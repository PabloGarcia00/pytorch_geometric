"""Disaggregate PV production from a P1 smart-meter stream using the
res_eval_cvae_5min_dualmask CVAE checkpoint, and compare the result against
earn-e's own processed PV estimate and the raw (non-interpolated) inverter
telemetry.

Model: dual-read (consumption_w/generation_w from the P1 meter's own
import/export registers -- no inverter data needed at inference), PV-only
target, 5-min cadence, 288-step (24h) lookback window per prediction.
See ../results/res_eval_cvae_5min_dualmask/config.yaml for the full config
this mirrors, and ../custom_graphgym/network/baseline_cvae_network.py for
the model this drives.

Usage:
    python disaggregate_pv.py --p1-id <p1-dongle-id> --eval-hours 24
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch

# Limits PyTorch's own intra-op thread pool -- unlike the OMP_NUM_THREADS/
# KMP_DUPLICATE_LIB_OK env vars (which must be set before numpy/pandas/torch
# are first imported to have any effect, see app.py's top), this is a
# runtime API call that's always effective regardless of import order. Part
# of the mitigation for a segfault observed specifically under a live
# `streamlit run` server (PyTorch's native thread pool racing with
# Streamlit's file-watcher thread) -- never reproduced standalone or under
# AppTest's single-threaded script runner. The model is tiny; single-
# threaded execution costs no meaningful latency.
# Read from OMP_NUM_THREADS (default 1) so one knob sizes both pools --
# docker-compose.yml raises it to match the container's CPU cap. Measured
# on the i7-8550U host: 1 thread 55s, 4 threads 20s for a full 1440-target
# batch (the BiLSTM is ~99% of it). The segfault above hasn't been seen
# at >1 thread yet, but it was never tested there either -- drop back to
# 1 if it reappears.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cache  # noqa: E402
import fetch_datahub as dh  # noqa: E402

import custom_graphgym  # noqa: E402, F401 -- registers baseline_cvae etc.
from custom_graphgym.eval.streaming_harness import WindowState  # noqa: E402
from custom_graphgym.loader.graph_dataset import _encode_temporal  # noqa: E402
from custom_graphgym.transform.transform import Transform  # noqa: E402

from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # noqa: E402
from torch_geometric.data.storage import GlobalStorage  # noqa: E402
from torch_geometric.graphgym.config import cfg, load_cfg  # noqa: E402
from torch_geometric.graphgym.model_builder import create_model  # noqa: E402

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

GRAPHGYM_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = GRAPHGYM_ROOT / "configs" / "pyg" / "res_eval_cvae_5min_dualmask.yaml"
CKPT_PATH = (
    GRAPHGYM_ROOT / "results" / "res_eval_cvae_5min_dualmask" / "0" / "ckpt"
    / "epoch=25-step=179374.ckpt"
)
TRANSFORM_PATH = (
    GRAPHGYM_ROOT / "datasets" / "earne_cvae_dual_physmask_5min" / "transform_dual.pt"
)

def _pred_cache_tag(mc_samples: int) -> str:
    """Keys cache.read_predictions/write_predictions. Bump the "steady-v1"
    suffix on any change to window construction or model inputs, so stale
    predictions are ignored rather than served. mc_samples is part of the
    tag too -- unlike the analytic path, Monte Carlo predictions
    (cfg.baseline.cvae_mc_samples > 1, see baseline_cvae_network.py's
    _mc_quantiles) are randomly drawn, so switching sample counts (or
    on/off) must not serve predictions cached under a different setting.
    """
    return f"{CKPT_PATH.name}|steady-v1|mc{mc_samples}"

SEQ_LEN = 288  # 24h @ 5min, fixed by the checkpoint's model.seq_len
FREQ = "5min"

log = logging.getLogger("disaggregate_pv")


def _load_state_dict(ckpt_path: Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if model_keys[0].startswith("model.") and not ckpt_keys[0].startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif ckpt_keys[0].startswith("model.") and not model_keys[0].startswith("model."):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def load_model() -> tuple[torch.nn.Module, torch.device]:
    cfg.set_new_allowed(True)
    load_cfg(cfg, argparse.Namespace(cfg_file=str(CFG_PATH), opts=[]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dim_in/dim_out passed explicitly -- skips create_dataset(), which would
    # otherwise try to rebuild the training dataset from the EFS-only gold
    # parquet that doesn't exist on this machine.
    model = create_model(dim_in=cfg.model.dim_in, dim_out=1, to_device=False)
    _load_state_dict(CKPT_PATH, model)
    model = model.to(device).eval()
    return model, device


def to_one_minute_grid(p1_df: pd.DataFrame) -> pd.DataFrame:
    """Strict 1-min Watts grid, small-gap-filled, with a per-tick presence
    flag (1 = genuine reading, 0 = gap-filled) -- the native-cadence feed
    for build_data_list's mixed-cadence windows (both the fine 5-slot tail
    and, median-aged, the coarse 283-slot history)."""
    df = p1_df.set_index("timeStamp").sort_index()
    raw = pd.DataFrame(
        {
            "consumption_w": df["netImportKw"] * 1000,
            "generation_w": df["netExportKw"] * 1000,
        }
    )
    grid = raw.resample("1min").median()
    present = grid["consumption_w"].notna().astype(float)
    grid = grid.ffill().bfill()
    grid["operational"] = present
    return grid.rename_axis("timestamp").reset_index()


# Ticks a WindowState must absorb before its seed has fully aged out:
# COARSE_LEN agings (FINE_LEN ticks each) refill `coarse`, and `fine` is
# all-real long before that. From then on the window is a pure function of
# the 1-min ticks -- independent of where pushing started.
WARMUP_TICKS = WindowState.COARSE_LEN * WindowState.FINE_LEN


def build_data_list(
    one_min: pd.DataFrame, transform: Transform, eval_start, eval_end, skip=frozenset(),
) -> tuple[list[Data], list]:
    """Mixed-cadence window per target: 283 coarse slots (each the median of
    5 real 1-min ticks, aged out of `fine`) + the 5 most-recent real 1-min
    ticks -- using custom_graphgym.eval.streaming_harness.WindowState for
    the coarse/fine aging mechanics rather than reimplementing the
    median-aging rule. One WindowState per channel (consumption,
    generation, operational) -- export is also 1-min-native at the P1
    meter, so it gets the same mixed-cadence treatment as import (unlike
    the original robustness study, where "generation" meant inverter data
    with a true 5-min ceiling).

    Steady state only: ticks are pushed from the start of `one_min` (the
    24h lookback before eval_start), and a target is emitted only once
    WARMUP_TICKS have flushed the placeholder seed. Pushing starts on a
    5-min clock boundary, so coarse groups are clock-aligned too. Together
    that makes each target's window -- and so its prediction -- depend only
    on the target time and the data, not on which eval window it was
    requested in. That's what lets run_disaggregation cache predictions
    per target (a seed taken at eval_start would change every past
    target's window each time a rolling live window moves).

    Targets in `skip` (already cached) still advance the windows but aren't
    materialized -- building each Data is the expensive part of this loop.
    """
    ticks = one_min[one_min["timestamp"] <= eval_end].sort_values("timestamp")
    ticks = ticks[ticks["timestamp"] >= ticks["timestamp"].iloc[0].ceil("5min")]

    # Placeholder seed -- fully aged out before any target is emitted.
    placeholder = [0.0] * WindowState.TOTAL_LEN
    w_cons = WindowState(placeholder)
    w_gen = WindowState(placeholder)
    w_op = WindowState(placeholder)

    # Parallel timestamp bookkeeping for the two deques WindowState itself
    # maintains (it tracks values only, not which real timestamp each slot
    # represents). `fine_ts` advances every tick; `coarse_ts` advances only
    # on the same tick WindowState's own aging fires (every FINE_LEN-th
    # push). to_one_minute_grid's output is gap-free, so the push count is
    # also the elapsed minutes.
    coarse_ts_dq = deque([pd.NaT] * WindowState.COARSE_LEN, maxlen=WindowState.COARSE_LEN)
    fine_ts_dq = deque([pd.NaT] * WindowState.FINE_LEN, maxlen=WindowState.FINE_LEN)

    data_list, target_timestamps = [], []
    for n_pushed, row in enumerate(ticks.itertuples(index=False), start=1):
        t = row.timestamp
        w_cons.push_tick(row.consumption_w)
        w_gen.push_tick(row.generation_w)
        w_op.push_tick(row.operational)

        fine_ts_dq.append(t)
        if n_pushed % WindowState.FINE_LEN == 0:
            coarse_ts_dq.append(t)

        # A prediction at every real 1-min tick, not just every 5th -- the
        # SM stream is 1-min-native and we want real-time nowcasting, not
        # 5-min-stratified output, even though the model's own history
        # cadence is 5-min.
        if n_pushed < WARMUP_TICKS or t < eval_start or t in skip:
            continue

        ts = pd.DatetimeIndex(list(coarse_ts_dq) + list(fine_ts_dq))

        cons_scaled = transform.transform("consumption", torch.tensor(w_cons.as_array(), dtype=torch.float32))
        gen_scaled = transform.transform("generation", torch.tensor(w_gen.as_array(), dtype=torch.float32))
        op = torch.tensor(w_op.as_array(), dtype=torch.float32)

        x = torch.stack([cons_scaled, gen_scaled], dim=-1).unsqueeze(0)  # [1, T, 2]
        operational = op.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        temporal = _encode_temporal(ts)  # [T, 6]

        data_list.append(
            Data(
                x=x,
                operational=operational,
                temporal=temporal,
                y_load=torch.zeros(1),
                y_pv=torch.zeros(1),
                y_net_demand=torch.zeros(1),
                mask=torch.zeros(1),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1)),
                num_nodes=1,
            )
        )
        target_timestamps.append(t)
    return data_list, target_timestamps


def run_inference(model, device, data_list, target_timestamps, transform) -> pd.DataFrame:
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        pred, _ = model(batch)
    pred_watts = transform.inverse_transform("pv", pred.cpu())
    cols = {
        f"pv_pred_q{int(q * 100)}_w": pred_watts[:, i].numpy()
        for i, q in enumerate(cfg.model.quantiles)
    }
    return pd.DataFrame({"timestamp": target_timestamps, **cols})


def merge_raw_solar(pred_df: pd.DataFrame, raw_df: pd.DataFrame, tolerance="30s") -> pd.DataFrame:
    """Raw inverter telemetry is sparse/irregular -- place each real reading
    at its single nearest prediction row (predictions are on a strict 1-min
    grid; tolerance="30s" is half that spacing, so a reading can match at
    most one row) and leave every other row NaN. No interpolation/fill and
    no spreading a single reading across multiple rows -- a wider tolerance
    (e.g. 5min) would nearest-match one real sample to every prediction
    timestamp within 5 minutes of it, smearing it across ~10 rows and
    visually hiding the genuine gaps between real readings."""
    raw = raw_df.rename(columns={"timeStamp": "timestamp"}).sort_values("timestamp").copy()
    raw["raw_solar_w"] = raw["powerOutputKw"] * 1000
    left = pred_df.sort_values("timestamp")
    return pd.merge_asof(
        left, raw[["timestamp", "raw_solar_w"]],
        on="timestamp", direction="nearest", tolerance=pd.Timedelta(tolerance),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p1-id", required=True, help="P1 dongle id (site) to disaggregate")
    p.add_argument("--eval-hours", type=int, default=24, help="how much history to predict PV for")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between API requests")
    p.add_argument(
        "--mc-samples", type=int, default=0,
        help="Monte Carlo predictive samples for uncertainty (0/1 = analytic quantiles, the original behavior)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def get_site(client, p1_id: str) -> dh.Site:
    """The model's only real prerequisite is the P1 (SM) signal -- inverters
    are optional and only needed for the ground-truth PV comparison, so a
    missing inverter is not an error here, just a reduced comparison later."""
    sites = dh.build_sites(cache.get_devices(client))
    site = next((s for s in sites if s.p1_id == p1_id), None)
    if site is None:
        raise ValueError(f"no site found for p1_id={p1_id!r}")
    return site


def run_disaggregation(
    client, model, device, transform, site: dh.Site, eval_start, end, sleep: float = 0.5, mc_samples: int = 0,
) -> pd.DataFrame:
    """Core pipeline: cached fetch -> resample -> windowed inference ->
    comparison against true PV. Shared by the CLI and the Streamlit app.

    True PV is the raw, non-interpolated inverter telemetry
    (raw_solar_graph.powerOutputKw) only -- earn-e's /graph endpoint
    deliberately doesn't expose a PV field, and a client-side derivation
    from buildingLoad/netImportKw/netExportKw turned out to lean on
    buildingLoad, which can itself go negative -- not a value a "load"
    should physically take, so not a reliable basis for ground truth. Raw
    solar is sparser and calibration-noisier, but it's an actual meter
    reading rather than a derived quantity.

    raw_solar_w is only fetched and populated when the site has an
    inverter -- otherwise it's left NaN throughout (comparison_metrics and
    the chart treat that as "omit", not an error).

    mc_samples: forwarded to cfg.baseline.cvae_mc_samples right before
    inference (not baked into load_model()/get_model()'s cached model
    object) -- cfg is a global yacs singleton the network reads live in
    forward(), so setting it once at model-load time and relying on
    st.cache_resource's per-argument caching would be wrong: whichever
    call set cfg last wins for every cached model, not just its own. 0/1
    disables (the original deterministic analytic-quantiles path); >1
    draws that many Monte Carlo predictive samples (see
    baseline_cvae_network.py's _mc_quantiles) -- same output shape either
    way, so nothing downstream needs to know which path ran.
    """
    cfg.baseline.cvae_mc_samples = mc_samples
    tag = _pred_cache_tag(mc_samples)

    # Floor to a 5-min boundary so the mixed-cadence target grid (built off
    # eval_start in build_data_list) stays in lockstep with WindowState's
    # 5-tick aging cadence -- "now" (today-mode in the app) isn't naturally
    # aligned to one.
    eval_start = pd.Timestamp(eval_start).as_unit("ns").floor("5min")
    end = pd.Timestamp(end).as_unit("ns")
    fetch_start = eval_start - pd.Timedelta(hours=24)  # SEQ_LEN lookback for the first window

    log.info("fetching P1 history (%s to %s)", fetch_start, end)
    p1_df = cache.fetch_cached(client, "p1_graph", client.get_graph, [site.p1_id], site.p1_id, fetch_start, end, 1, sleep)
    one_min = to_one_minute_grid(p1_df)

    # Per-target prediction cache: build_data_list's windows are steady-state
    # (see its docstring), so a target's prediction never changes once its
    # minute is complete -- a live tick only runs the model on the new tail.
    cached = cache.read_predictions(site.p1_id, tag, eval_start, end)
    data_list, target_ts = build_data_list(one_min, transform, eval_start, end, skip=set(cached["timestamp"]))
    if data_list:
        log.info("running inference on %d target timesteps (%d cached)", len(data_list), len(cached))
        new_df = run_inference(model, device, data_list, target_ts, transform)
        # The newest 1-min tick may still be missing readings the API hasn't
        # served yet (the next fetch picks them up) -- only cache targets
        # whose own minute is complete, so the latest one gets recomputed.
        complete = new_df[new_df["timestamp"] < end.floor("1min")]
        cache.write_predictions(site.p1_id, tag, complete)
        pred_df = pd.concat([cached, new_df], ignore_index=True) if len(cached) else new_df
    elif len(cached):
        pred_df = cached
    else:
        raise ValueError("no target windows in eval range -- insufficient P1 history fetched")
    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)

    if site.inverter_ids:
        log.info("fetching raw solar telemetry (%s to %s)", eval_start, end)
        raw_solar_df = cache.fetch_cached(client, "raw_solar_graph", client.get_raw_solar_graph, site.inverter_ids, site.p1_id, eval_start, end, 1, sleep)
        merged = merge_raw_solar(pred_df, raw_solar_df)
    else:
        log.info("site has no inverters -- no true PV available, comparison column left blank")
        merged = pred_df.copy()
        merged["raw_solar_w"] = float("nan")

    # 1-min-native display columns, matching the prediction grid's own
    # cadence.
    merged = merged.merge(one_min[["timestamp", "consumption_w", "generation_w"]], on="timestamp", how="left")
    return merged


def comparison_metrics(merged: pd.DataFrame) -> tuple[float, int]:
    has_raw = merged["raw_solar_w"].notna()
    mae_raw = (
        (merged.loc[has_raw, "pv_pred_q50_w"] - merged.loc[has_raw, "raw_solar_w"]).abs().mean()
        if has_raw.any() else float("nan")
    )
    return mae_raw, int(has_raw.sum())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end = datetime.now(timezone.utc)
    eval_start = end - timedelta(hours=args.eval_hours)

    client = dh.DatahubClient(dh.load_api_key())
    site = get_site(client, args.p1_id)

    log.info("loading model + transform")
    model, device = load_model()
    transform = Transform.load(TRANSFORM_PATH)

    merged = run_disaggregation(client, model, device, transform, site, eval_start, end, args.sleep, args.mc_samples)

    out_path = args.out_dir / site.p1_id / "pv_disaggregation_comparison.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    mae_raw, n_raw = comparison_metrics(merged)
    log.info("wrote %s (%d rows)", out_path, len(merged))
    log.info(
        "MAE vs true PV / raw solar telemetry (%d matched samples): %s",
        n_raw, f"{mae_raw:.1f} W" if n_raw else "n/a",
    )


if __name__ == "__main__":
    main()
