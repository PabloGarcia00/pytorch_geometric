"""
trajectory.py

Visual confirmation of disaggregation quality: overlays each config's
predicted load/PV trajectory (median + q_lo-q_hi band) against one shared
real trajectory, for a single household and time window.

Sweeps `results/` generically via notebooks/calculate_metrics.py's
discover_manifest()/disaggregate_test_set() — no per-grid config parsing to
maintain; any sweep directory with `-key=value-...` run names works.

Fair-comparison design:
    - The real trajectory is drawn once; every compared run's prediction is
      overlaid on it, so no config gets a favorable/unfavorable duplicate of
      the ground truth.
    - Predictions are averaged across a run's seeds (same config, same test
      split — only initialization/training stochasticity differs).
    - The household is picked by pooling masked MAE across *all* compared
      runs, not just one — so it isn't cherry-picked to flatter or hurt any
      single config.
    - Runs that don't predict the requested target are skipped with a
      warning rather than silently misreported.

Usage:
    python visualization/trajectory.py \\
        --sweep earne_learned_corr_single_grid_sweep_coverage_weather \\
        --target pv --household auto:median --days 7 --out trajectory.pdf
"""

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import custom_graphgym  # noqa — registers loaders/losses/metrics
from calculate_metrics import CACHE_FIT_MODEL_TYPES, discover_manifest, disaggregate_test_set
from custom_graphgym.metric.regression import DisaggregationMetrics
from custom_graphgym.target_utils import active_targets
from torch_geometric.graphgym.config import cfg

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
STEPS_PER_DAY = 96  # fixed 15-min cadence, project-wide (global.cadence_minutes)

TRUE_IDX = {"load": DisaggregationMetrics.TRUE_LOAD, "pv": DisaggregationMetrics.TRUE_PV}


# --- 1. LATEX / THESIS STYLE CONFIG ---
def init_style():
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.bbox": "tight",
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
    })


COLOR_MAP = ["#ECCD61", "#7FA3C2", "#B88FD5", "#34495e", "#5FAD8C", "#D97C6B"]


# --- 2. INFERENCE ---
def _run_predictions(row: pd.Series, target: str):
    """
    Run inference for one (run, seed) row and slice out `target`'s q50/q_lo/
    q_hi trajectories plus the shared real/mask/timestamps.

    Returns None if this row doesn't predict `target`, or has no checkpoint,
    or its dataset isn't available on this machine.
    """
    if target not in row["predict_targets"]:
        return None

    result = disaggregate_test_set(row)
    if result is None:
        return None
    true, pred, user_ids, timestamps = result

    # cfg still reflects `row`'s config here — disaggregate_test_set's
    # load_cfg call mutated the module-level singleton and nothing has
    # touched it since (synchronous, same pattern as calculate_metrics.py).
    targets = active_targets()
    quantiles = list(cfg.model.quantiles)
    n_q = cfg.model.n_quantiles
    block = targets.index(target)
    q50_idx = quantiles.index(0.5)
    qlo_idx, qhi_idx = int(np.argmin(quantiles)), int(np.argmax(quantiles))

    pred_block = pred[:, block * n_q : (block + 1) * n_q, :].numpy()  # [N, n_q, T]

    return {
        "q50": pred_block[:, q50_idx, :],
        "qlo": pred_block[:, qlo_idx, :],
        "qhi": pred_block[:, qhi_idx, :],
        "real": true[:, TRUE_IDX[target], :].numpy(),
        "mask": true[:, DisaggregationMetrics.TRUE_MASK, :].bool().numpy(),
        "timestamps": timestamps,
        "user_ids": list(user_ids),
    }


def aggregate_run(rows: pd.DataFrame, target: str):
    """
    rows: manifest rows for every seed of one run_name.
    Averages q50/q_lo/q_hi across seeds (same config, same test split — only
    initialization differs). real/mask/timestamps/user_ids are identical
    across seeds by construction, so they're taken from the first one.
    """
    per_seed = [
        r for r in (_run_predictions(row, target) for _, row in rows.iterrows())
        if r is not None
    ]
    if not per_seed:
        return None

    ref = per_seed[0]
    return {
        "q50": np.mean([p["q50"] for p in per_seed], axis=0),
        "qlo": np.mean([p["qlo"] for p in per_seed], axis=0),
        "qhi": np.mean([p["qhi"] for p in per_seed], axis=0),
        "real": ref["real"],
        "mask": ref["mask"],
        "timestamps": ref["timestamps"],
        "user_ids": ref["user_ids"],
        "n_seeds": len(per_seed),
    }


# --- 3. HOUSEHOLD + WINDOW SELECTION ---
def select_household(run_results: dict, mode: str) -> str:
    """
    Picks a household present in every compared run, ranked by mean masked
    MAE pooled *across all runs* — fair in that no single config's
    performance determines which household gets shown.
    """
    common_ids = set(next(iter(run_results.values()))["user_ids"])
    for res in run_results.values():
        common_ids &= set(res["user_ids"])
    if not common_ids:
        raise ValueError("No household is present across every compared run.")

    scores = {}
    for uid in common_ids:
        errs = []
        for res in run_results.values():
            i = res["user_ids"].index(uid)
            m = res["mask"][i]
            if not m.any():
                continue
            errs.append(np.abs(res["q50"][i][m] - res["real"][i][m]).mean())
        if errs:
            scores[uid] = float(np.mean(errs))

    if not scores:
        raise ValueError("No household has any valid (unmasked) test timesteps.")

    ranked = sorted(scores, key=scores.get)
    if mode == "best":
        return ranked[0]
    if mode == "worst":
        return ranked[-1]
    if mode == "median":
        return ranked[len(ranked) // 2]
    raise ValueError(f"Unknown household selection mode: {mode!r}")


def find_window_start(timestamps: list, days: int) -> int:
    """First local-midnight-aligned index with `days` full days remaining."""
    steps = days * STEPS_PER_DAY
    for i, t in enumerate(timestamps):
        local = t.astimezone(AMSTERDAM)
        if local.hour == 0 and local.minute == 0 and i + steps <= len(timestamps):
            return i
    return 0  # no clean local-midnight boundary found — fall back to the start


# --- 4. PLOTTING ---
def plot_household_trajectory(
    run_results: dict, uid: str, target: str, days: int, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))

    ref = next(iter(run_results.values()))
    timestamps = ref["timestamps"]
    steps = days * STEPS_PER_DAY
    start = find_window_start(timestamps, days)
    end = min(start + steps, len(timestamps))
    window_ts = [t.astimezone(AMSTERDAM) for t in timestamps[start:end]]

    real_drawn = False
    for i, (run_name, res) in enumerate(sorted(run_results.items())):
        j = res["user_ids"].index(uid)
        mask = res["mask"][j, start:end]
        real = np.where(mask, res["real"][j, start:end], np.nan)
        q50 = res["q50"][j, start:end]
        qlo = res["qlo"][j, start:end]
        qhi = res["qhi"][j, start:end]

        if not real_drawn:
            ax.plot(window_ts, real, color="#222222", linewidth=1.3, label="Real", zorder=10)
            real_drawn = True

        color = COLOR_MAP[i % len(COLOR_MAP)]
        n_seeds = res["n_seeds"]
        label = f"{run_name} (pred, n={n_seeds})"
        ax.plot(window_ts, q50, color=color, linewidth=1.0, label=label)
        ax.fill_between(window_ts, qlo, qhi, color=color, alpha=0.15, linewidth=0)

    ax.set_ylabel(f"{target.upper()} (W)")
    fig.supxlabel("Time")
    fig.autofmt_xdate()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncols=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    print(
        f"[✓] Saved -> {out_path}  "
        f"(household={uid}, target={target}, "
        f"window={window_ts[0].date()} -> {window_ts[-1].date()})"
    )


# --- 5. MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay predicted-vs-real disaggregation trajectories across a results sweep."
    )
    parser.add_argument("--sweep", required=True, help="sweep dir name under results/")
    parser.add_argument("--runs", default=None, help="optional substring filter on run_name")
    parser.add_argument("--target", choices=["load", "pv"], required=True)
    parser.add_argument(
        "--household",
        default="auto:median",
        help="'auto:best' | 'auto:median' | 'auto:worst' | an explicit user_id",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", default="trajectory.pdf")
    args = parser.parse_args()

    init_style()
    manifest = discover_manifest()
    sub = manifest[manifest["sweep"] == args.sweep]
    if args.runs:
        sub = sub[sub["run_name"].str.contains(args.runs)]
    sub = sub[sub["ckpt_path"].notna() | sub["model_type"].isin(CACHE_FIT_MODEL_TYPES)]
    if sub.empty:
        raise SystemExit(f"No runnable rows found for sweep={args.sweep!r} runs={args.runs!r}")

    run_results = {}
    for run_name, rows in sub.groupby("run_name"):
        res = aggregate_run(rows, args.target)
        if res is None:
            print(f"  skip {run_name}: doesn't predict {args.target!r}, or no checkpoints/dataset")
            continue
        run_results[run_name] = res

    if not run_results:
        raise SystemExit(f"No run in sweep={args.sweep!r} predicts target={args.target!r}")

    if args.household.startswith("auto:"):
        uid = select_household(run_results, args.household.split(":", 1)[1])
    else:
        uid = args.household

    plot_household_trajectory(run_results, uid, args.target, args.days, args.out)
