"""
export_3panel_3day_stitched.py

3-panel figure, one model per panel (CVAE / MLP / KNN). Each panel plots a
STITCHED 288-step (96x3) profile: 3 back-to-back day-segments, each segment
from a potentially different (household, date) pair. The *set* of 3
(household, date) segments is selected once, pooled across all 3 models,
and reused identically in every panel -- so panel-to-panel, you're looking
at the exact same 3 segments, only the model changes. Segments are NOT
required to be temporally contiguous in real calendar time (a segment can
be household 58's May day, the next household 56's a different month) --
the x-axis is a stitched step index (0-287) with vertical dividers and a
household/date label per segment, not one continuous real timeline.

Selection, in order:
  1. 3 households: median-RMSE window (select_households_by_rmse_window),
     pooled across CVAE/MLP/KNN, from already-scored
     notebooks/metrics_per_household.csv -- no new inference needed.
  2. Per household, one day: median-RMSE local-midnight-aligned day, RMSE
     pooled across CVAE/MLP/KNN's predictions for THAT household (needs
     each model's full per-timestep trajectory, restricted via filter_ids
     to just these 3 households -- one inference call per model, not nine).

Usage:
    CUDA_VISIBLE_DEVICES="" python -u visualization/export_3panel_3day_stitched.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

import calculate_metrics
from calculate_metrics import discover_manifest
from trajectory import AMSTERDAM, MODEL_CONFIGS, STEPS_PER_DAY, load_named_run

_real_load_cfg = calculate_metrics.load_cfg
_target_uids: list[str] = []

# IMPORTANT: MLP/LSTM/GNN all share cfg.model.type == "earne_network" --
# GraphGym registers them under one shared wrapper class, and the actual
# architecture is selected by cfg.dataset.node_encoder_name instead
# ("baseline_mlp_temporal" / "baseline_lstm_temporal" / "earne_temporal").
# Checking cfg.model.type alone (as an earlier version of this script did)
# silently treated MLP/LSTM as "GNN-like" and never restricted them --
# they ran on the full population every time, which is harmless for
# correctness by itself (MLP/LSTM have no cross-household state) but
# turned out to matter for a DIFFERENT reason: it meant their dataset
# build could end up on a different code path than CVAE/KNN's restricted
# builds, and this script was (wrongly) reusing CVAE's integer day-start
# index directly as an index into every other model's own timestamps
# array without checking they were actually date-aligned. Fixed below by
# looking up each model's own day-start index by calendar DATE, not by
# reusing a raw integer position across models.
_RESTRICTABLE = {
    ("baseline_mlp", None), ("baseline_lstm", None),
    ("baseline_cvae", None), ("baseline_knn", None),
    ("earne_network", "baseline_mlp_temporal"),
    ("earne_network", "baseline_lstm_temporal"),
}
_t_last = [time.monotonic()]


def _patched_load_cfg(cfg_obj, args):
    _real_load_cfg(cfg_obj, args)
    node_enc = getattr(cfg_obj.model, "node_encoder_name", None)
    key_specific = (cfg_obj.model.type, node_enc)
    key_generic = (cfg_obj.model.type, None)
    restricted = bool(_target_uids and (key_specific in _RESTRICTABLE or key_generic in _RESTRICTABLE))
    if restricted:
        cfg_obj.earne_data.filter_ids = list(_target_uids)
    now = time.monotonic()
    print(
        f"  -> loading {cfg_obj.model.type}/{node_enc} (restricted={restricted}) [+{now - _t_last[0]:.1f}s]",
        flush=True,
    )
    _t_last[0] = now


calculate_metrics.load_cfg = _patched_load_cfg

PANEL_MODELS = ["CVAE", "MLP", "KNN"]
DUAL_NOMASK_WX_OFF_RUNS = {
    "CVAE": "baseline_cvae_dual_nomask-wx=False",
    "MLP": "baseline_mlp_dual_nomask-wx=False",
    "KNN": "baseline_knn_dual-wx=False",
}


def select_3_households(models: list[str], n: int = 3) -> list[str]:
    df = pd.read_csv("notebooks/metrics_per_household.csv", low_memory=False)
    run_names = [DUAL_NOMASK_WX_OFF_RUNS[m] for m in models]
    sub = df[(df["run_name"].isin(run_names)) & (df["target"] == "pv") & (df["metric"] == "rmse")]
    per_run_household = sub.groupby(["run_name", "user_id"])["value"].mean().reset_index()
    piv = per_run_household.pivot(index="user_id", columns="run_name", values="value")
    common = piv.dropna()
    if len(common) < n:
        raise SystemExit(f"Only {len(common)} common households, need {n}.")
    pooled = common.mean(axis=1).sort_values()
    mid = len(pooled) // 2
    half = n // 2
    start = max(0, min(mid - half, len(pooled) - n))
    return [str(u) for u in pooled.index[start:start + n].tolist()]


def day_start_indices(timestamps: list) -> list[int]:
    starts = []
    for i, t in enumerate(timestamps):
        local = t.astimezone(AMSTERDAM)
        if local.hour == 0 and local.minute == 0 and i + STEPS_PER_DAY <= len(timestamps):
            starts.append(i)
    return starts


def run() -> None:
    global _target_uids
    manifest = discover_manifest()

    uids = select_3_households(PANEL_MODELS)
    print(f"3 households (median-RMSE window, pooled across {PANEL_MODELS}): {uids}")

    # One restricted (to just these 3 households) full-trajectory inference
    # call per model -- gives every household's full test-split trajectory
    # for that model in one pass, needed both for pooled day-selection and
    # for the final stitched panels.
    results = {}
    for label in PANEL_MODELS:
        _target_uids = uids
        sweep, run_prefix = MODEL_CONFIGS[label]["dual"]
        run_name = f"{run_prefix}-wx=False"
        res = load_named_run(manifest, sweep, run_name, "pv", best_seed=True)
        _target_uids = []
        if res is None:
            raise SystemExit(f"No runnable result for {label} / {sweep}/{run_name}")
        results[label] = res
        print(f"[{label}] trajectory loaded, {len(res['timestamps'])} timesteps, households={res['user_ids']}")

    # Per-model day-start-index lookup, keyed by calendar date -- looked up
    # independently per model rather than assuming a shared integer index
    # is valid across models. Different models can (and did, for MLP vs
    # CVAE here) end up on dataset builds that are the same LENGTH but not
    # date-aligned index-for-index, so reusing one model's raw index into
    # another's array silently picks the wrong day. This is checked
    # explicitly below, not just assumed.
    date_to_start = {}
    for label in PANEL_MODELS:
        ts = results[label]["timestamps"]
        m = {}
        for s in day_start_indices(ts):
            d = ts[s].astimezone(AMSTERDAM).date().isoformat()
            m.setdefault(d, s)
        date_to_start[label] = m

    ref_label = PANEL_MODELS[0]
    ref_ts = results[ref_label]["timestamps"]
    misaligned = [
        label for label in PANEL_MODELS[1:]
        if len(results[label]["timestamps"]) != len(ref_ts)
        or results[label]["timestamps"][0] != ref_ts[0]
    ]
    if misaligned:
        print(
            f"NOTE: {misaligned} dataset(s) are not date-aligned with {ref_label}'s "
            f"(different length and/or start timestamp) -- this is handled by looking "
            f"up each segment's date independently per model below, not by reusing a "
            f"shared integer index, so results stay correct regardless."
        )

    starts = day_start_indices(ref_ts)
    if not starts:
        raise SystemExit("No local-midnight-aligned full day found.")

    # Per household: median-RMSE day, RMSE pooled across all 3 models'
    # predictions for that one household. Candidate days come from the
    # reference model's calendar dates; each model's OWN start index for
    # that same date is looked up via date_to_start (falls back to
    # skipping a model for that day if it has no data on that date at all).
    segments = []  # (uid, date_str)
    for uid in uids:
        day_rmses = []
        for s in starts:
            date_str = ref_ts[s].astimezone(AMSTERDAM).date().isoformat()
            errs = []
            for label in PANEL_MODELS:
                s_label = date_to_start[label].get(date_str)
                if s_label is None:
                    continue
                res = results[label]
                i = res["user_ids"].index(uid)
                e = s_label + STEPS_PER_DAY
                m = res["mask"][i][s_label:e]
                if not m.any():
                    continue
                real = res["real"][i][s_label:e][m]
                q50 = res["q50"][i][s_label:e][m]
                errs.append(float(np.sqrt(((real - q50) ** 2).mean())))
            if len(errs) == len(PANEL_MODELS):  # only consider days every model has data for
                day_rmses.append((date_str, float(np.mean(errs))))
        if not day_rmses:
            raise SystemExit(f"Household {uid}: no date has valid data from every model.")
        day_rmses.sort(key=lambda x: x[1])
        date_str = day_rmses[len(day_rmses) // 2][0]
        segments.append((uid, date_str))
        print(f"  household {uid}: median-RMSE day = {date_str} (pooled across {PANEL_MODELS})")

    # Stitch: for each model, concatenate the 3 segments (each segment uses
    # THAT model's own prediction for its assigned household+date, with
    # that model's OWN start index for that date -- not a shared index).
    panels = []
    for label in PANEL_MODELS:
        res = results[label]
        true_pv, q50, qlo, qhi = [], [], [], []
        for uid, date_str in segments:
            i = res["user_ids"].index(uid)
            s = date_to_start[label][date_str]
            e = s + STEPS_PER_DAY
            true_pv += res["real"][i][s:e].tolist()
            q50 += res["q50"][i][s:e].tolist()
            qlo += res["qlo"][i][s:e].tolist()
            qhi += res["qhi"][i][s:e].tolist()
        panels.append({
            "model": label,
            "true_pv": true_pv, "pv_q50": q50, "pv_qlo": qlo, "pv_qhi": qhi,
            "seed": res["seed"], "test_loss": res["test_loss"],
        })

    payload = {
        "config": "dual_wx_off",
        "steps_per_day": STEPS_PER_DAY,
        "segments": [{"household": uid, "date": date} for uid, date in segments],
        "segment_rule": (
            "3 households: median pooled RMSE window across "
            f"{PANEL_MODELS} (metrics_per_household.csv). Per household: "
            f"median-RMSE local-midnight-aligned day, pooled across {PANEL_MODELS}. "
            "Same 3 (household, date) segments reused identically in every panel/model."
        ),
        "panels": panels,
    }
    out_path = Path("visualization/data/models_3panel_3day_stitched.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    run()
