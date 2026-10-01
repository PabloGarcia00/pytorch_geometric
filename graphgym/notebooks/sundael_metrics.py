"""Score the Sundael physics disaggregator's PV predictions with the SAME
metrics, window, and output tables as the graphgym models, so it drops into
notebooks/metrics_*.csv and notebooks/held_out_metrics_*.csv alongside them.

Sundael isn't a graphgym run (no config.yaml / checkpoint / manifest row), so
calculate_metrics.py's discover_manifest() can't pick it up. This script
instead reads the generation-prediction parquets that scratch_sundael_transfer.py
produced (seen + held-out populations), rebuilds the gold-layer ground truth
over graphgym's val+test target window exactly as the loader does, and reuses
calculate_metrics.py's scoring (DisaggregationMetrics) and atomic scoped-merge
CSV writer so the sundael rows are appended without disturbing any other model's.

Scored like the other models: PV target only (every existing metrics_long row
is target=='pv'), val+test window combined (season-balanced full year, see
calculate_metrics.disaggregate_test_set), point estimate (n_quantiles=1 — no
quantile head, so coverage/sharpness are degenerate and ~0 by construction).

Run (graphgym venv), after scratch_sundael_transfer.py has produced predictions:
    .venv/bin/python notebooks/sundael_metrics.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import polars as pl

from calculate_metrics import (  # noqa: E402 — reuse the production scoring + IO
    METRIC_LABELS,
    _merge_and_write,
    _metrics_to_records,
    _strat_fields,
    prune_metrics,
)
from custom_graphgym.metric.regression import DisaggregationMetrics  # noqa: E402

# ── config — mirrors the 15-min dual-read CVAE config scratch_sundael_transfer.py
# reproduces (and that the bulk of the fleet is scored under) ────────────────
GOLD = "/home/llan/projects/messm/pytorch_geometric/fleet_gold_layer.parquet"
PRED_DIR = Path(__file__).resolve().parent.parent / "results" / "sundael_transfer"
START = dt.datetime(2023, 4, 3, tzinfo=dt.timezone.utc)
END = START.replace(year=2026)
SEQ_LEN = 96
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.85
ACTIVITY_COLS = ["consumption_w", "generation_w", "inverter_w", "load_w"]

SWEEP = "sundael_transfer"
SEED = 0
# strat fields for the synthetic manifest rows — physics disaggregator, no
# trainable encoder/head/loss; dual-read P1 registers; temperature+clearsky
# geometry as its weather anchor; 15-min native resolution (shows "native"
# like the other 15-min models). require_full_span=True describes the model's
# config intent in both tables (as held_out_metrics.py records it too), not
# the population being scored.
SUNDAEL_STRAT = {
    "model_type": "sundael",
    "node_encoder_name": None,
    "head_name": None,
    "loss_fun": None,
    "dual_read": True,
    "weather_mode": "physics",
    "graph_mode": None,
    "mask_physics_impossible": None,
    "physics_weight": None,
    "require_full_span": True,
    "resolution": "native",
    "tstr_protocol": None,
    "synthetic_source": None,
}

# (run_name, seen-prediction file, held-out-prediction file)
RUNS = [
    ("sundael_clearsky", "pred_generation_clearsky.parquet", "pred_generation_heldout_clearsky.parquet"),
    ("sundael_ref", "pred_generation_ref.parquet", "pred_generation_heldout_ref.parquet"),
]


def log(*a):
    print("[sundael-metrics]", *a, flush=True)


# ── gold-layer ground truth (matches graph_dataset._process_shared) ─────────
def load_gold() -> dict:
    """Rebuild master ids, timestamps, the val+test target window, and the
    PV/load/net/mask arrays over ALL native users — exactly as
    custom_graphgym.loader.graph_dataset does (same date span, activity-trim,
    0-fill, all-activity-columns-present mask, and 70/85 chronological split)."""
    lf = (
        pl.scan_parquet(GOLD)
        .with_columns(pl.col("user_id").cast(pl.Utf8))
        .filter(pl.col("timestamp").is_between(START, END, closed="none"))
        .filter(pl.col("zipcode").is_not_null())
    )
    last_ts = (
        lf.filter(pl.any_horizontal(pl.col(ACTIVITY_COLS).abs() > 0))
        .select(pl.col("timestamp").max())
        .collect()
        .item()
    )
    df = lf.filter(pl.col("timestamp") <= last_ts).sort(["timestamp", "user_id"]).collect()

    ids = sorted(df["user_id"].unique().to_list())
    ts = pd.to_datetime(sorted(df["timestamp"].unique().to_list()), utc=True)
    log(f"gold: {len(ids)} native users x {len(ts)} timesteps")

    def piv_raw(col):
        return (
            df.pivot(values=col, index="timestamp", on="user_id")
            .sort("timestamp")
            .select(ids)
            .to_numpy()
        )  # [T, N], NaN where missing

    raw = {c: piv_raw(c) for c in ACTIVITY_COLS + ["net_demand_w"]}
    mask = np.stack([~np.isnan(raw[c]) for c in ACTIVITY_COLS]).all(0)  # [T, N]
    # 0-fill the label channels (graph_dataset._pivot_filled); masked-out
    # entries are excluded by `mask` downstream so the fill value never scores.
    pv = np.nan_to_num(raw["inverter_w"])
    load = np.nan_to_num(raw["load_w"])
    net = np.nan_to_num(raw["net_demand_w"])

    # val+test target timesteps, mirroring _calc_splits + disaggregate_test_set:
    # eval_idx = val+test sample indices; each target column is ts[idx + seq_len].
    n_samples = len(ts) - SEQ_LEN - 1
    train_end = int(n_samples * TRAIN_SPLIT)
    eval_idx = list(range(train_end, n_samples))  # val then test, ascending
    tgt = [i + SEQ_LEN for i in eval_idx]
    target_ts = ts[tgt]
    log(f"val+test window: {len(tgt)} target timesteps "
        f"[{target_ts[0]} .. {target_ts[-1]}]")

    return dict(
        ids=ids, target_ts=target_ts, tgt=np.asarray(tgt),
        pv=pv, load=load, net=net, mask=mask,
    )


def build_true(G: dict, users: list[str]) -> tuple[np.ndarray, list[str]]:
    """true [N, 4, T] = (load, pv, mask, net_demand) over the val+test window
    for `users`, in DisaggregationMetrics' channel order.

    Users with no supervised timestep in the val+test window (e.g. a held-out
    household active only in the train period) are dropped — they contribute
    nothing to pooled scoring and break per-household scoring (empty arrays)."""
    col = {u: i for i, u in enumerate(G["ids"])}
    cidx = [col[u] for u in users]
    tgt = G["tgt"]
    load = G["load"][np.ix_(tgt, cidx)].T  # [N, T]
    pv = G["pv"][np.ix_(tgt, cidx)].T
    net = G["net"][np.ix_(tgt, cidx)].T
    mask = G["mask"][np.ix_(tgt, cidx)].T.astype(np.float32)

    keep = mask.sum(axis=1) > 0
    if not keep.all():
        dropped = [u for u, k in zip(users, keep) if not k]
        log(f"  dropping {len(dropped)} user(s) with no supervised val+test data: {dropped}")
        users = [u for u, k in zip(users, keep) if k]
        load, pv, net, mask = load[keep], pv[keep], net[keep], mask[keep]

    true = np.stack([load, pv, mask, net], axis=1)  # [N, 4, T]
    return true, users


def build_pred(pred_df: pd.DataFrame, users: list[str], target_ts) -> np.ndarray:
    """pred [N, 1, T] — sundael's point PV (generation) estimate, aligned to
    `users` x the val+test target timestamps."""
    pred_df = pred_df.copy()
    pred_df.index = pred_df.index.astype(str)
    pred_df.columns = pd.to_datetime(pred_df.columns, utc=True)
    aligned = pred_df.reindex(index=users, columns=target_ts)
    return aligned.to_numpy()[:, None, :]  # [N, 1, T]


# Sundael is a single point estimate — no predictive interval. The quantile/
# interval metrics DisaggregationMetrics still emits (with q10==q50==q90) are
# degenerate and misleading next to models with real bands (e.g. coverage
# collapses to the night-time all-zeros match rate), so they're NaN'd out. The
# point metrics (rmse/mae/r2/efe/mbe/export_violation + pinball_q50) stand.
_INTERVAL_METRICS = frozenset({
    "coverage", "sharpness", "nsharpness",
    "pinball_q10", "npinball_q10", "pinball_q90", "npinball_q90",
})


def _null_interval_metrics(metrics: dict) -> dict:
    for tm in metrics.values():
        for m in _INTERVAL_METRICS & tm.keys():
            tm[m] = float("nan")
    return metrics


def score(true: np.ndarray, pred: np.ndarray, users: list[str]) -> tuple[dict, list[dict]]:
    """Pooled + per-household PV metrics (point estimate: n_quantiles=1)."""
    pooled = _null_interval_metrics(DisaggregationMetrics.all(
        true, pred, targets=["pv"], n_quantiles=1, q50_idx=0,
    ))
    household = {}
    for i, uid in enumerate(users):
        household[uid] = _null_interval_metrics(DisaggregationMetrics.all(
            true[i : i + 1], pred[i : i + 1], targets=["pv"], n_quantiles=1, q50_idx=0,
        ))
    return pooled, household


def _row(run_name: str) -> pd.Series:
    return pd.Series({"sweep": SWEEP, "run_name": run_name, "seed": SEED, **SUNDAEL_STRAT})


def _household_records(row: pd.Series, household: dict) -> list[dict]:
    strat = _strat_fields(row)
    recs = []
    for uid, metrics in household.items():
        for target, tm in metrics.items():
            for metric, value in tm.items():
                recs.append({
                    "sweep": row["sweep"], "run_name": row["run_name"], "seed": row["seed"],
                    **strat, "user_id": uid, "target": target, "metric": metric, "value": value,
                })
    return recs


def _regen_wide(long_path: str, wide_path: str) -> None:
    """Rebuild the mean-aggregated wide table from the merged long CSV — same
    pivot/prune/label as calculate_metrics.py's __main__."""
    merged = pd.read_csv(long_path, low_memory=False)
    wide = merged.pivot_table(
        index="metric", columns=["target", "sweep", "run_name"], values="value", aggfunc="mean",
    )
    wide = prune_metrics(wide).rename(index=METRIC_LABELS)
    wide.to_csv(wide_path)


def main():
    G = load_gold()
    members = json.loads((PRED_DIR / "sundael_membership.json").read_text())
    log(f"membership: {len(members['seen_ids'])} seen, {len(members['heldout_ids'])} held-out")

    long_records, hh_records = [], []
    ho_long_records, ho_hh_records = [], []

    for order, (run_name, seen_file, heldout_file) in enumerate(RUNS):
        row = _row(run_name)

        # ── seen population → metrics_long / metrics_per_household ──
        pred_df = pd.read_parquet(PRED_DIR / seen_file)
        users = [u for u in members["seen_ids"] if str(u) in set(pred_df.index.astype(str))]
        true, users = build_true(G, users)
        pred = build_pred(pred_df, users, G["target_ts"])
        pooled, household = score(true, pred, users)
        log(f"[{run_name}] seen: {len(users)} users | "
            f"PV nmae={pooled['pv']['nmae']:.3f} r2={pooled['pv']['r2']:.3f} rmse={pooled['pv']['rmse']:.1f}W")
        long_records.extend(_metrics_to_records(row, pooled, order))
        hh_records.extend(_household_records(row, household))

        # ── held-out population → held_out_metrics_long / _per_household ──
        ho_path = PRED_DIR / heldout_file
        if ho_path.exists():
            ho_pred_df = pd.read_parquet(ho_path)
            ho_users = [u for u in members["heldout_ids"] if str(u) in set(ho_pred_df.index.astype(str))]
            if ho_users:
                ho_true, ho_users = build_true(G, ho_users)
                ho_pred = build_pred(ho_pred_df, ho_users, G["target_ts"])
                ho_pooled, ho_household = score(ho_true, ho_pred, ho_users)
                log(f"[{run_name}] held-out: {len(ho_users)} users | "
                    f"PV nmae={ho_pooled['pv']['nmae']:.3f} r2={ho_pooled['pv']['r2']:.3f} "
                    f"rmse={ho_pooled['pv']['rmse']:.1f}W")
                ho_long_records.extend(_metrics_to_records(row, ho_pooled, order))
                ho_hh_records.extend(_household_records(row, ho_household))
            else:
                log(f"[{run_name}] held-out: no scorable users, skipping")
        else:
            log(f"[{run_name}] held-out predictions {ho_path.name} missing, skipping")

    # ── write (scoped merge: add/replace only sundael rows, keep every other
    # model's; same atomic locked writer the production scripts use) ──
    def write(long_recs, hh_recs, long_csv, hh_csv, wide_csv):
        if not long_recs:
            return
        df_long = (
            pd.DataFrame(long_recs)
            .sort_values(["_order", "target", "metric"]).drop(columns="_order")
        )
        _merge_and_write(df_long, long_csv, scoped=True)
        _regen_wide(long_csv, wide_csv)
        log(f"[OK] wrote {long_csv} (+{len(df_long)} sundael rows) and {wide_csv}")
        if hh_recs:
            df_hh = pd.DataFrame(hh_recs).sort_values(
                ["sweep", "run_name", "seed", "user_id", "target", "metric"]
            )
            _merge_and_write(df_hh, hh_csv, scoped=True)
            log(f"[OK] wrote {hh_csv} (+{len(df_hh)} sundael rows)")

    write(long_records, hh_records,
          "notebooks/metrics_long.csv", "notebooks/metrics_per_household.csv",
          "notebooks/metrics_wide.csv")
    write(ho_long_records, ho_hh_records,
          "notebooks/held_out_metrics_long.csv", "notebooks/held_out_metrics_per_household.csv",
          "notebooks/held_out_metrics_wide.csv")


if __name__ == "__main__":
    main()
