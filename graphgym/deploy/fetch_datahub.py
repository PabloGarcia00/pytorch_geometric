"""Fetch P1 smart-meter, inverter, and raw solar data from the earn-e datahub.

Exploration-phase pull for the disaggregation pipeline (see ../ for the dev
environment that consumes this data). Talks to the three endpoints documented
in example_code.md:

  - GET  /historic/getdevices     -> site -> {p1_id, inverter_ids}
  - POST /historic/graph          -> aggregated load/solar timeseries.
                                      Requires the P1 id in deviceIds; an
                                      inverter-only request 400s with
                                      "Nodetype not recognized".
  - POST /historic/rawsolargraph  -> raw (non-interpolated) solar output,
                                      inverter id(s) only, max 24h span/request

Per site this produces two files: p1_graph (P1 alone) and combined_graph
(P1 + inverters together, i.e. load net of solar), plus raw_solar_graph.

Usage:
    python fetch_datahub.py --start 2026-09-14 --end 2026-09-21

Writes into ./data/ by default (this dir) — never into the disaggregation
dev environment above, to avoid silently contaminating that dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# pandas 3.0 defaults to PyArrow-backed StringDtype inference for string
# columns. Confirmed via a live-browser + faulthandler reproduction (not
# guessed) that pd.DataFrame(rows) below -- rows is a list of dicts from
# the raw API JSON, with `timeStamp` as the one string column -- segfaults
# inside pandas.core.arrays.string_arrow._from_sequence when run under
# Streamlit's live server. Never reproduced standalone or under AppTest,
# only via an actual browser session driving the real Tornado server --
# consistent with a concurrency-triggered bug in pandas 3.0.3's new Arrow
# string path. Disabling the inference reverts to the mature, battle-tested
# object-dtype string path, avoiding that code entirely.
pd.set_option("future.infer_string", False)

BASE_URL = "https://datahub.earn-e.com/api/v1/historic"
ENV_PATH = Path(__file__).resolve().parent / ".env"

UUID_LIKE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]+$")

log = logging.getLogger("fetch_datahub")


def load_api_key(env_path: Path = ENV_PATH) -> str:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "API_KEY":
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"API_KEY not found in {env_path}")


def classify_device(device_id: str) -> str:
    """P1 dongle IDs are dashed UUID-style strings; inverter IDs are plain hex."""
    return "p1" if "-" in device_id else "inverter"


@dataclass
class Site:
    p1_id: str
    inverter_ids: list[str]


def build_sites(raw_devices: list[list[str]]) -> list[Site]:
    sites = []
    for row in raw_devices:
        p1_ids = [d for d in row if classify_device(d) == "p1"]
        inverter_ids = [d for d in row if classify_device(d) == "inverter"]
        if not p1_ids:
            log.warning("device row with no P1 id, skipping: %s", row)
            continue
        sites.append(Site(p1_id=p1_ids[0], inverter_ids=inverter_ids))
    return sites


class DatahubClient:
    def __init__(self, api_key: str, timeout: float = 30.0):
        self.session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "hub-api-key": api_key,
            }
        )
        self.timeout = timeout

    def get_devices(self) -> list[list[str]]:
        resp = self.session.get(f"{BASE_URL}/getdevices", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_graph(self, device_ids: list[str], start_ms: int, end_ms: int, samples: int) -> list[dict]:
        payload = {
            "deviceIds": [{"id": d} for d in device_ids],
            "start": start_ms,
            "end": end_ms,
            "samples": samples,
        }
        resp = self.session.post(f"{BASE_URL}/graph", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["graph"]

    def get_raw_solar_graph(self, device_ids: list[str], start_ms: int, end_ms: int, samples: int) -> list[dict]:
        payload = {
            "deviceIds": [{"id": d} for d in device_ids],
            "start": start_ms,
            "end": end_ms,
            "samples": samples,
        }
        resp = self.session.post(f"{BASE_URL}/rawsolargraph", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["graph"]


def day_chunks(start: datetime, end: datetime, span: timedelta = timedelta(hours=24)):
    cur = start
    while cur < end:
        nxt = min(cur + span, end)
        yield cur, nxt
        cur = nxt


def fetch_ranged(
    client: DatahubClient,
    fetch_fn,
    device_ids: list[str],
    start: datetime,
    end: datetime,
    interval_minutes: int,
    sleep_s: float,
) -> pd.DataFrame:
    """Chunk [start, end) into <=24h windows and concatenate results."""
    frames = []
    for chunk_start, chunk_end in day_chunks(start, end):
        start_ms = int(chunk_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000) - 1
        minutes = (chunk_end - chunk_start).total_seconds() / 60
        samples = max(1, int(minutes / interval_minutes))
        rows = fetch_fn(device_ids, start_ms, end_ms, samples)
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timeStamp" in df.columns:
        # Force a fixed resolution here, at the one place raw API timestamps
        # get parsed -- pandas' flexible-resolution inference can otherwise
        # pick 'ms'/'us'/'ns' depending on the JSON string's precision, which
        # then mismatches Timestamps built elsewhere (e.g. from a bare date)
        # and breaks comparisons/merges downstream.
        df["timeStamp"] = pd.to_datetime(df["timeStamp"], utc=True).astype("datetime64[ns, UTC]")
        df = df.sort_values("timeStamp").drop_duplicates("timeStamp").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if df.empty:
        log.warning("no data for %s, skipping write", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)
    log.info("wrote %s (%d rows)", path.with_suffix(f".{fmt}"), len(df))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=str, default=None, help="ISO date/time (UTC), default: 1 day ago")
    p.add_argument("--end", type=str, default=None, help="ISO date/time (UTC), default: now")
    p.add_argument("--interval-minutes", type=int, default=1, help="sample cadence, default 1min")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--p1-id", action="append", default=None, help="restrict to these P1 site(s); repeatable")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between API requests")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else end - timedelta(days=1)

    client = DatahubClient(load_api_key())

    log.info("fetching device list")
    raw_devices = client.get_devices()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "_devices.json").write_text(json.dumps(raw_devices, indent=2))

    sites = build_sites(raw_devices)
    if args.p1_id:
        sites = [s for s in sites if s.p1_id in args.p1_id]
    log.info("found %d site(s)%s", len(sites), " (filtered)" if args.p1_id else "")

    for site in sites:
        site_dir = args.out_dir / site.p1_id
        log.info("site %s: p1=%s inverters=%s", site.p1_id, site.p1_id, site.inverter_ids)

        p1_df = fetch_ranged(client, client.get_graph, [site.p1_id], start, end, args.interval_minutes, args.sleep)
        save(p1_df, site_dir / "p1_graph", args.format)

        if site.inverter_ids:
            # /graph rejects inverter-only device lists ("Nodetype not recognized") -
            # it needs the P1 id present to aggregate; this is the combined
            # household load + solar view, not a standalone inverter graph.
            combined_ids = [site.p1_id, *site.inverter_ids]
            combined_df = fetch_ranged(client, client.get_graph, combined_ids, start, end, args.interval_minutes, args.sleep)
            save(combined_df, site_dir / "combined_graph", args.format)

            solar_df = fetch_ranged(client, client.get_raw_solar_graph, site.inverter_ids, start, end, args.interval_minutes, args.sleep)
            save(solar_df, site_dir / "raw_solar_graph", args.format)
        else:
            log.warning("site %s has no inverters, skipping combined/solar pulls", site.p1_id)


if __name__ == "__main__":
    main()
