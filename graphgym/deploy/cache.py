"""Read-through SQLite cache in front of the earn-e datahub API.

Wraps fetch_datahub.py's client so repeated requests (e.g. a Streamlit app
re-running the same site/date-range, or a live-updating dashboard polling a
rolling window that shifts forward each tick) don't re-hit the live API for
data already on disk. Incremental: a (p1_id, stream, start, end) request
only fetches whatever sub-range isn't already covered (usually just the new
tail for a polling caller), not the whole range every time.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

import fetch_datahub as dh

DB_PATH = Path(__file__).resolve().parent / "data" / "cache.sqlite"

# stream name -> (data columns beyond p1_id/ts, source dataframe timestamp column)
# combined_graph (P1+inverter /graph, incl. buildingLoad/selfConsumptionKw) is
# deliberately not cached/fetched -- buildingLoad can go negative (confirmed
# on real data), which undermines any PV estimate derived from it. Raw solar
# telemetry is the only true-PV source this pipeline trusts.
_STREAMS = {
    "p1_graph": ["avgImportKw", "avgExportKw", "totalGas", "interpolatedPowerKw", "netImportKw", "netExportKw"],
    "raw_solar_graph": ["powerOutputKw"],
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL")
    for stream, cols in _STREAMS.items():
        col_defs = ", ".join(f'"{c}" REAL' for c in cols)
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{stream}" '
            f'(p1_id TEXT NOT NULL, ts TEXT NOT NULL, {col_defs}, '
            f"PRIMARY KEY (p1_id, ts))"
        )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS coverage ("
        "p1_id TEXT NOT NULL, stream TEXT NOT NULL, "
        "start_ts TEXT NOT NULL, end_ts TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS devices (raw_json TEXT NOT NULL, fetched_at TEXT NOT NULL)"
    )
    # Long format (one row per prediction column) so the table doesn't
    # depend on the checkpoint's quantile set -- model_tag scopes that.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "p1_id TEXT NOT NULL, model_tag TEXT NOT NULL, ts TEXT NOT NULL, "
        "col TEXT NOT NULL, value REAL, "
        "PRIMARY KEY (p1_id, model_tag, ts, col))"
    )
    return conn


def _missing_ranges(
    conn: sqlite3.Connection, p1_id: str, stream: str, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Gaps between [start, end] and the union of existing coverage rows --
    i.e. what actually needs fetching. A polling caller (e.g. a live-updating
    dashboard requesting a rolling window that shifts forward each tick)
    only pays for the new tail, not the whole window every time."""
    rows = conn.execute(
        "SELECT start_ts, end_ts FROM coverage WHERE p1_id=? AND stream=? "
        "AND end_ts >= ? AND start_ts <= ? ORDER BY start_ts",
        (p1_id, stream, start.isoformat(), end.isoformat()),
    ).fetchall()
    missing = []
    cur = start
    for s, e in rows:
        s, e = pd.Timestamp(s).as_unit("ns"), pd.Timestamp(e).as_unit("ns")
        if s > cur:
            missing.append((cur, s))
        cur = max(cur, e)
        if cur >= end:
            return missing
    if cur < end:
        missing.append((cur, end))
    return missing


def _merge_coverage(conn: sqlite3.Connection, p1_id: str, stream: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    rows = conn.execute(
        "SELECT rowid, start_ts, end_ts FROM coverage WHERE p1_id=? AND stream=?",
        (p1_id, stream),
    ).fetchall()
    merged_start, merged_end = start, end
    to_delete = []
    for rowid, s, e in rows:
        s, e = pd.Timestamp(s).as_unit("ns"), pd.Timestamp(e).as_unit("ns")
        if s <= merged_end and e >= merged_start:  # overlaps or touches
            merged_start, merged_end = min(merged_start, s), max(merged_end, e)
            to_delete.append(rowid)
    if to_delete:
        conn.executemany("DELETE FROM coverage WHERE rowid=?", [(r,) for r in to_delete])
    conn.execute(
        "INSERT INTO coverage (p1_id, stream, start_ts, end_ts) VALUES (?, ?, ?, ?)",
        (p1_id, stream, merged_start.isoformat(), merged_end.isoformat()),
    )


def _read_range(conn: sqlite3.Connection, stream: str, p1_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cols = _STREAMS[stream]
    col_list = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql_query(
        f'SELECT ts, {col_list} FROM "{stream}" WHERE p1_id=? AND ts>=? AND ts<=? ORDER BY ts',
        conn, params=(p1_id, start.isoformat(), end.isoformat()),
    )
    # SQLite round-trips timestamps as ISO text. Two issues fixed here:
    # (1) datetime.isoformat() omits the fractional-seconds part entirely
    #     when microsecond==0, so the "ts" column can mix "...06+00:00" and
    #     "...06.450000+00:00" rows -- format="ISO8601" handles that mix
    #     (a fixed-format parse errors out on it).
    # (2) to_datetime otherwise infers whatever sub-second resolution the
    #     text happens to carry, which can silently mismatch a caller's
    #     'ns'/'s'-resolution Timestamps and break comparisons/merges --
    #     force a single resolution at this boundary.
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
    return df.rename(columns={"ts": "timeStamp"})


def _write_rows(conn: sqlite3.Connection, stream: str, p1_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = _STREAMS[stream]
    rows = [
        (p1_id, ts.isoformat(), *[row.get(c) for c in cols])
        for ts, row in zip(df["timeStamp"], df.to_dict("records"))
    ]
    placeholders = ", ".join(["?"] * (2 + len(cols)))
    col_list = ", ".join(f'"{c}"' for c in cols)
    conn.executemany(
        f'INSERT OR REPLACE INTO "{stream}" (p1_id, ts, {col_list}) VALUES ({placeholders})',
        rows,
    )


def get_devices(client: "dh.DatahubClient", max_age_hours: float = 24.0) -> list[list[str]]:
    """Cached getdevices -- refetched if the cached copy is older than max_age_hours."""
    import json
    with closing(_connect()) as conn:
        row = conn.execute("SELECT raw_json, fetched_at FROM devices ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is not None:
            raw_json, fetched_at = row
            if pd.Timestamp.now(tz="UTC") - pd.Timestamp(fetched_at) < pd.Timedelta(hours=max_age_hours):
                return json.loads(raw_json)
        devices = client.get_devices()
        conn.execute(
            "INSERT INTO devices (raw_json, fetched_at) VALUES (?, ?)",
            (json.dumps(devices), pd.Timestamp.now(tz="UTC").isoformat()),
        )
        conn.commit()
        return devices


def fetch_cached(
    client: "dh.DatahubClient",
    stream: str,
    fetch_fn,
    device_ids: list[str],
    p1_id: str,
    start,
    end,
    interval_minutes: int = 1,
    sleep_s: float = 0.5,
) -> pd.DataFrame:
    """Read-through cache for one of the datahub streams (see _STREAMS).

    `stream` keys the cache table (p1_graph/raw_solar_graph). `p1_id` is the
    cache partition key (one site); `device_ids` is what's actually posted
    to the API (inverter ids for raw_solar_graph -- caching is per-site, not
    per-device-id-set, since a site's device composition is stable).
    """
    # Normalize to a fixed resolution -- callers may pass a Timestamp built
    # from a bare datetime.date (resolves to 's') or from parsed API JSON
    # (resolves to 'ns'/'us'); comparing/merging mixed resolutions is the
    # bug this guards against.
    start, end = pd.Timestamp(start).as_unit("ns"), pd.Timestamp(end).as_unit("ns")
    with closing(_connect()) as conn:
        missing = _missing_ranges(conn, p1_id, stream, start, end)
        for m_start, m_end in missing:
            df = dh.fetch_ranged(client, fetch_fn, device_ids, m_start, m_end, interval_minutes, sleep_s)
            _write_rows(conn, stream, p1_id, df)
            _merge_coverage(conn, p1_id, stream, m_start, m_end)
        if missing:
            conn.commit()
        return _read_range(conn, stream, p1_id, start, end)


def read_predictions(p1_id: str, model_tag: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Cached per-target predictions in [start, end], wide (timestamp + one
    column per prediction column) -- see disaggregate_pv.run_disaggregation
    for why a target's prediction is stable enough to cache."""
    with closing(_connect()) as conn:
        df = pd.read_sql_query(
            "SELECT ts, col, value FROM predictions WHERE p1_id=? AND model_tag=? AND ts>=? AND ts<=?",
            conn, params=(p1_id, model_tag, start.isoformat(), end.isoformat()),
        )
    if df.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
    wide = df.pivot(index="ts", columns="col", values="value").rename_axis(columns=None).reset_index()
    # Same ISO-text round-trip normalization as _read_range.
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
    return wide.rename(columns={"ts": "timestamp"})


def write_predictions(p1_id: str, model_tag: str, pred_df: pd.DataFrame) -> None:
    cols = [c for c in pred_df.columns if c != "timestamp"]
    rows = [
        (p1_id, model_tag, ts.isoformat(), c, float(v))
        for ts, *vals in pred_df[["timestamp", *cols]].itertuples(index=False)
        for c, v in zip(cols, vals)
    ]
    if not rows:
        return
    with closing(_connect()) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO predictions (p1_id, model_tag, ts, col, value) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
