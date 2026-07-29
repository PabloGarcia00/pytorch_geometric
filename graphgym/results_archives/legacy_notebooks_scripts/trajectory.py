# trajectory of PV and load
#
# two experiments:
#  - seasonal: winter, summer, shoulder
#    (11, 12, 1, 2)
#    (5, 6, 7, 8) T                WHEN extract('month' from {ts_col}) IN (11, 12, 1, 2) THEN 'Winter'
#    else shoudler
#  - active_diverse_selection

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset
from custom_graphgym.loader.normalization import denorm_ihs, denorm_log1p
from custom_graphgym.metric.regression import DisaggregationMetrics

sys.path.insert(0, "/path/to/graphgym")


# ── config ────────────────────────────────────────────────────────────────────
OUTPUT_DATA = Path(
    "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/output/earne_exp3_mean_only"
)
DENORM_PARAMS = torch.load(
    "datasets/earne/norm_params_log1p_ihs_znorm_spatial_knn.pt"
)


EXP_LABELS = {
    "earne_exp3_mean_only-weather_mode=False-mp=0": r"W- MP0",
    "earne_exp3_mean_only-weather_mode=False-mp=3": r"W- MP3",
    "earne_exp3_mean_only-weather_mode=True-mp=0": r"W+ MP0",
    "earne_exp3_mean_only-weather_mode=True-mp=3": r"W+ MP3",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def denorm_pred(pred, params):
    load_denorm = denorm_log1p(pred[..., :3, :], *params["load_scaled"])
    pv_denorm = denorm_ihs(pred[..., 3:, :], *params["pv_scaled"])
    return torch.cat([load_denorm, pv_denorm], dim=1)


def denorm_true(true, params):
    load_denorm = denorm_log1p(true[..., [0], :], *params["load_scaled"])
    pv_denorm = denorm_ihs(true[..., [1], :], *params["pv_scaled"])
    net_denorm = denorm_ihs(true[..., [3], :], *params["net_scaled"])
    mask = true[..., [2], :]
    return torch.cat([load_denorm, pv_denorm, mask, net_denorm], dim=1)


def flatten_metrics(metrics: dict) -> dict:
    """{'load': {'rmse': 1.0, ...}, 'pv': {...}} → {'load_rmse': 1.0, ...}"""
    return {
        f"{target}_{metric}": value
        for target, m in metrics.items()
        for metric, value in m.items()
    }


# ── pair pred/true files by experiment name ───────────────────────────────────

pred_files = {
    p.name.replace("_pred.pt", ""): p for p in OUTPUT_DATA.glob("*pred.pt")
}
true_files = {
    p.name.replace("_true.pt", ""): p for p in OUTPUT_DATA.glob("*true.pt")
}
experiments = sorted(pred_files.keys() & true_files.keys())

print(experiments)

pred = denorm_pred(torch.load(pred_files[experiments[1]]), DENORM_PARAMS)
true = denorm_true(torch.load(true_files[experiments[1]]), DENORM_PARAMS)

# ── 3. Recreate dataset + test loader ────────────────────────────────────────
input_data = EARNeGraphDataset("datasets/earne")
_, _, test_idx = input_data.get_split_indices()

STEPS_PER_DAY = 96  # 15-min intervals
STEPS_PER_WEEK = 96 * 7

test_timestamps = np.array(input_data.timestamps)[test_idx]

AMS = ZoneInfo("Europe/Amsterdam")

# test_timestamps = np.array([t.astimezone(AMS) for t in test_timestamps])

# ── 1. find clean monday→sunday window ───────────────────────────────────────
first_monday = next(
    i
    for i, t in enumerate(test_timestamps)
    if t.weekday() == 0 and t.hour == 0 and t.minute == 0
)
# last complete week: trim tail so length is divisible by STEPS_PER_WEEK
n_steps = len(test_timestamps) - first_monday
n_weeks = n_steps // STEPS_PER_WEEK
clip = first_monday + n_weeks * STEPS_PER_WEEK

# same for days
first_day = next(
    i for i, t in enumerate(test_timestamps) if t.hour == 0 and t.minute == 0
)
n_days = (len(test_timestamps) - first_day) // STEPS_PER_DAY
clip_day = first_day + n_days * STEPS_PER_DAY

print(
    f"weeks: {n_weeks}  ({test_timestamps[first_monday].date()} → {test_timestamps[clip-1].date()})"
)
print(
    f"days:  {n_days}  ({test_timestamps[first_day].date()} → {test_timestamps[clip_day-1].date()})"
)

# ── unpack and mask true channels ─────────────────────────────────────────────
mask = true[:, 2, :].bool()  # [N, T]
load = true[:, 0, :].float().masked_fill(~mask, float("nan"))  # [N, T]
pv = true[:, 1, :].float().masked_fill(~mask, float("nan"))  # [N, T]
net = true[:, 3, :].float().masked_fill(~mask, float("nan"))  # [N, T]

# ── slice to clean windows ────────────────────────────────────────────────────
load_w = load[:, first_monday:clip].contiguous()
pv_w = pv[:, first_monday:clip].contiguous()
net_w = net[:, first_monday:clip].contiguous()
mask_w = mask[:, first_monday:clip].contiguous()

load_d = load[:, first_day:clip_day].contiguous()
pv_d = pv[:, first_day:clip_day].contiguous()
net_d = net[:, first_day:clip_day].contiguous()
mask_d = mask[:, first_day:clip_day].contiguous()

# ── reshape into blocks ───────────────────────────────────────────────────────
N = true.shape[0]

load_w = load_w.reshape(N * n_weeks, STEPS_PER_WEEK)
pv_w = pv_w.reshape(N * n_weeks, STEPS_PER_WEEK)
net_w = net_w.reshape(N * n_weeks, STEPS_PER_WEEK)
mask_w = mask_w.reshape(N * n_weeks, STEPS_PER_WEEK)

load_d = load_d.reshape(N * n_days, STEPS_PER_DAY)
pv_d = pv_d.reshape(N * n_days, STEPS_PER_DAY)
net_d = net_d.reshape(N * n_days, STEPS_PER_DAY)
mask_d = mask_d.reshape(N * n_days, STEPS_PER_DAY)


# ── variance per block ────────────────────────────────────────────────────────
def profile_score(load, pv, mask, pv_mean_weight=5.0, pv_var_weight=2.0):
    """
    Combines mean PV (solar activity) + PV variance (dynamics) + load variance.
    Profiles with flat but nonzero PV will rank higher than purely spiky ones.
    """
    valid_frac = mask.float().mean(dim=1).numpy()

    load_var = np.nanvar(load.numpy(), axis=1)
    pv_var = np.nanvar(pv.numpy(), axis=1)
    pv_mean = np.nanmean(pv.numpy(), axis=1)  # ← key addition

    scores = load_var + pv_var_weight * pv_var + pv_mean_weight * pv_mean
    scores[valid_frac < 0.05] = np.nan
    return scores


week_scores = profile_score(load_w, pv_w, mask_w)
day_scores = profile_score(load_d, pv_d, mask_d)

print(f"week: {np.isfinite(week_scores).sum()} / {len(week_scores)} valid rows")
print(f"day:  {np.isfinite(day_scores).sum()}  / {len(day_scores)}  valid rows")

# sort by variance
week_order = np.argsort(week_scores)
day_order = np.argsort(day_scores)

N_SAMPLES = 5
selections = {
    "high": {
        "week": week_order[-N_SAMPLES:][::-1],
        "day": day_order[-N_SAMPLES:][::-1],
    },
    "median": {
        "week": week_order[
            len(week_order) // 2
            - N_SAMPLES // 2 : len(week_order) // 2
            + N_SAMPLES // 2
        ],
        "day": day_order[
            len(day_order) // 2
            - N_SAMPLES // 2 : len(day_order) // 2
            + N_SAMPLES // 2
        ],
    },
    "low": {"week": week_order[:N_SAMPLES], "day": day_order[:N_SAMPLES]},
}

# ── unpack pred channels ──────────────────────────────────────────────────────
# pred: [N, 6, T] — load_q10, load_q50, load_q90, pv_q10, pv_q50, pv_q90
load_q10 = pred[:, 0, :].float()
load_q50 = pred[:, 1, :].float()
load_q90 = pred[:, 2, :].float()
pv_q10 = pred[:, 3, :].float()
pv_q50 = pred[:, 4, :].float()
pv_q90 = pred[:, 5, :].float()


# ── slice + reshape weekly ────────────────────────────────────────────────────
def slice_reshape(x, start, end, n_blocks, steps):
    return x[:, start:end].contiguous().reshape(n_blocks * N, steps)


load_q10_w = slice_reshape(
    load_q10, first_monday, clip, n_weeks, STEPS_PER_WEEK
)
load_q50_w = slice_reshape(
    load_q50, first_monday, clip, n_weeks, STEPS_PER_WEEK
)
load_q90_w = slice_reshape(
    load_q90, first_monday, clip, n_weeks, STEPS_PER_WEEK
)
pv_q10_w = slice_reshape(pv_q10, first_monday, clip, n_weeks, STEPS_PER_WEEK)
pv_q50_w = slice_reshape(pv_q50, first_monday, clip, n_weeks, STEPS_PER_WEEK)
pv_q90_w = slice_reshape(pv_q90, first_monday, clip, n_weeks, STEPS_PER_WEEK)

# ── slice + reshape daily ─────────────────────────────────────────────────────
load_q10_d = slice_reshape(load_q10, first_day, clip_day, n_days, STEPS_PER_DAY)
load_q50_d = slice_reshape(load_q50, first_day, clip_day, n_days, STEPS_PER_DAY)
load_q90_d = slice_reshape(load_q90, first_day, clip_day, n_days, STEPS_PER_DAY)
pv_q10_d = slice_reshape(pv_q10, first_day, clip_day, n_days, STEPS_PER_DAY)
pv_q50_d = slice_reshape(pv_q50, first_day, clip_day, n_days, STEPS_PER_DAY)
pv_q90_d = slice_reshape(pv_q90, first_day, clip_day, n_days, STEPS_PER_DAY)


# ── error score per row ───────────────────────────────────────────────────────
def error_scores(true_load, true_pv, pred_load_q50, pred_pv_q50, mask):
    m = mask.bool()
    valid_frac = m.float().mean(dim=1)
    load_err = (pred_load_q50 - true_load).abs().masked_fill(~m, float("nan"))
    pv_err = (pred_pv_q50 - true_pv).abs().masked_fill(~m, float("nan"))
    scores = torch.from_numpy(
        np.nanmean(load_err.numpy(), axis=1)
    ) + torch.from_numpy(np.nanmean(pv_err.numpy(), axis=1))
    scores[valid_frac < 0.05] = float("nan")
    return scores


week_errors = error_scores(load_w, pv_w, load_q50_w, pv_q50_w, mask_w)
day_errors = error_scores(load_d, pv_d, load_q50_d, pv_q50_d, mask_d)


# ── joint sort: variance tier + error rank ────────────────────────────────────
def joint_selection(var_scores, err_scores, n_blocks, n=N_SAMPLES):
    var_order = np.argsort(var_scores)
    n_total = len(var_scores)
    tiers = {
        "high": var_order[-n_total // 3 :][::-1].copy(),
        "median": var_order[n_total // 3 : 2 * n_total // 3].copy(),
        "low": var_order[: n_total // 3].copy(),
    }
    result = {}
    for tier, idxs in tiers.items():
        err_in_tier = err_scores[idxs]
        err_order = np.argsort(err_in_tier.numpy())
        result[tier] = {
            "best": idxs[err_order[:n]],
            "worst": idxs[err_order[-n:]],
        }
    return result


week_selection = joint_selection(week_scores, week_errors, n_weeks)
day_selection = joint_selection(day_scores, day_errors, n_days)
week_selection

# ── decode and print ──────────────────────────────────────────────────────────
for tier, v in week_selection.items():
    for quality, idxs in v.items():
        print(f"\nweek {tier} variance / {quality} error:")
        for idx in idxs:
            node, block = idx // n_weeks, idx % n_weeks
            date = test_timestamps[first_monday + block * STEPS_PER_WEEK]
            print(
                f"  node={node:3d}  week={date.date()}  "
                f"var={week_scores[idx]:.1f}  err={week_errors[idx]:.1f}"
            )

# ── Export ──────────────────────────────────────────────────────────


def extract_sample(
    flat_idx,
    n_blocks,
    steps,
    granularity,
    load_true,
    pv_true,
    net_true,
    load_q10,
    load_q50,
    load_q90,
    pv_q10,
    pv_q50,
    pv_q90,
    var_scores,
    err_scores,
    first_start,
    test_timestamps,
):
    node = flat_idx // n_blocks
    block = flat_idx % n_blocks
    start = first_start + block * steps

    ts = [t.isoformat() for t in test_timestamps[start : start + steps]]

    return {
        "node": int(node),
        "block": int(block),
        "date": test_timestamps[start].date().isoformat()[:10],
        "var_score": float(var_scores[flat_idx]),
        "err_score": float(err_scores[flat_idx]),
        "timestamps": ts,
        "true": {
            "load": load_true[flat_idx].tolist(),
            "pv": pv_true[flat_idx].tolist(),
            "net": net_true[flat_idx].tolist(),
        },
        "pred": {
            "load_q10": load_q10[flat_idx].tolist(),
            "load_q50": load_q50[flat_idx].tolist(),
            "load_q90": load_q90[flat_idx].tolist(),
            "pv_q10": pv_q10[flat_idx].tolist(),
            "pv_q50": pv_q50[flat_idx].tolist(),
            "pv_q90": pv_q90[flat_idx].tolist(),
        },
    }


def build_export(
    selection,
    n_blocks,
    steps,
    granularity,
    load_true,
    pv_true,
    net_true,
    load_q10,
    load_q50,
    load_q90,
    pv_q10,
    pv_q50,
    pv_q90,
    var_scores,
    err_scores,
    first_start,
    test_timestamps,
    max_total=10,
):

    # flatten all selected indices to enforce global cap
    all_idxs = [
        (tier, quality, idx)
        for tier, qualities in selection.items()
        for quality, idxs in qualities.items()
        for idx in idxs
    ]

    out = {}
    total = 0
    for tier, qualities in selection.items():
        out[tier] = {}
        for quality, idxs in qualities.items():
            if total >= max_total:
                out[tier][quality] = []
                continue
            out[tier][quality] = [
                extract_sample(
                    int(idx),
                    n_blocks,
                    steps,
                    granularity,
                    load_true,
                    pv_true,
                    net_true,
                    load_q10,
                    load_q50,
                    load_q90,
                    pv_q10,
                    pv_q50,
                    pv_q90,
                    var_scores,
                    err_scores,
                    first_start,
                    test_timestamps,
                )
                for idx in idxs
            ]
    return out


def interpolate(arr, method="time"):
    s = pd.Series(arr)
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy()


def interpolate_blocks(tensor_2d):
    arr = tensor_2d.numpy()
    return np.stack([interpolate(row) for row in arr])


data = (
    build_export(
        week_selection,
        n_weeks,
        STEPS_PER_WEEK,
        "week",
        interpolate_blocks(load_w),
        interpolate_blocks(pv_w),
        interpolate_blocks(net_w),
        load_q10_w,
        load_q50_w,
        load_q90_w,
        pv_q10_w,
        pv_q50_w,
        pv_q90_w,
        week_scores,
        week_errors,
        first_monday,
        test_timestamps,
        5,
    ),
)
data[0].keys()
for key in data[0].keys():
    print(data[0][key].keys())
    print(data[0][key].keys())
data[0]["low"]["worst"]

# ── build and export ──────────────────────────────────────────────────────────
export = {
    "week": build_export(
        week_selection,
        n_weeks,
        STEPS_PER_WEEK,
        "week",
        interpolate_blocks(load_w),
        interpolate_blocks(pv_w),
        interpolate_blocks(net_w),
        load_q10_w,
        load_q50_w,
        load_q90_w,
        pv_q10_w,
        pv_q50_w,
        pv_q90_w,
        week_scores,
        week_errors,
        first_monday,
        test_timestamps,
        5,
    ),
    "day": build_export(
        day_selection,
        n_days,
        STEPS_PER_DAY,
        "day",
        interpolate_blocks(load_d),
        interpolate_blocks(pv_d),
        interpolate_blocks(net_d),
        load_q10_d,
        load_q50_d,
        load_q90_d,
        pv_q10_d,
        pv_q50_d,
        pv_q90_d,
        day_scores,
        day_errors,
        first_day,
        test_timestamps,
        5,
    ),
}

with open("profiles.json", "w") as f:
    json.dump(export, f, indent=2)

print("saved → profiles.json")
print(
    f"  weeks: {sum(len(v) for tier in export['week'].values() for v in tier.values())} samples"
)
print(
    f"  days:  {sum(len(v) for tier in export['day'].values()  for v in tier.values())} samples"
)
