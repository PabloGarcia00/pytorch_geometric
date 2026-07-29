"""
trajectory.py

Visual confirmation of disaggregation quality: overlays predicted PV/load
trajectories (median + optional q_lo-q_hi band) against one shared real
trajectory, for a single household and time window.

Sweeps `results/` generically via notebooks/calculate_metrics.py's
discover_manifest()/disaggregate_test_set() — no per-grid config parsing to
maintain. Comparisons can span multiple sweep dirs (multiple models), via
load_named_run() + the MODEL_SWEEPS registry, not just configs within one
sweep dir.

Three modes:
    --mode models (default): every model in MODEL_SWEEPS, same no-mask/
        no-loss/weather-off condition, one figure -- "how do all 7 models
        disaggregate this same household/window."
    --mode physics: one model (--model), three tiers -- no mask/loss, mask
        only (pw=0.0), mask+loss (pw=0.3) -- household/window chosen for a
        bad PV<export violation under the no-mask/no-loss tier, with the
        physical export floor + violation counts overlaid.
    --mode mae: legacy single-sweep comparison (household ranked by pooled
        MAE across whatever configs share one --sweep dir) -- kept for
        ad-hoc debugging, not the recommended path for either figure above.

Fair-comparison design:
    - The real trajectory is drawn once; every compared run's prediction is
      overlaid on it, so no config gets a favorable/unfavorable duplicate of
      the ground truth.
    - Predictions are averaged across a run's seeds (same config, same test
      split — only initialization/training stochasticity differs).
    - Runs that don't predict the requested target are skipped with a
      warning rather than silently misreported.

Usage:
    python visualization/trajectory.py --mode models --days 2 --out models.pdf
    python visualization/trajectory.py --mode physics --model MLP --days 2 --out physics_mlp.pdf
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

# Model registry for --mode models / --mode physics. Per model: a sweep+run-prefix for
# each read mode (weather on/off is a run-name suffix within that same sweep dir, not a
# separate sweep), plus the physics-mask sweep (dual-read only -- no single-read physics
# experiment exists). Fixed from this project's actual sweep-dir names (not derivable by
# a generic template -- naming isn't fully uniform across models, e.g. "_nomask" only
# appears for the gradient-trained-encoder baselines, not KNN/Linear/SVR/ST-GNN).
#
# "ST-GNN" is a *display-name-only* rename of the earne_network model (environment-wide
# convention going forward, per user request) -- the underlying sweep/run-dir strings
# below still say "earne_learned_corr..." on disk and are left as-is; renaming those
# would require touching every existing config/results path referencing them for no
# benefit (they're an internal detail, never shown to a reader of a figure).
MODEL_CONFIGS = {
    "ST-GNN": {
        "dual": ("earne_learned_corr_dual_grid_sweep_weather_only", "earne_learned_corr_dual"),
        "single": ("earne_learned_corr_single_csi_grid_sweep_weather_only", "earne_learned_corr_single_csi"),
        "physmask": "earne_learned_corr_dual_physmask_grid_sweep_physics_weight",
    },
    "MLP": {
        "dual": ("baseline_mlp_dual_nomask_grid_sweep_weather_only", "baseline_mlp_dual_nomask"),
        "single": ("baseline_mlp_single_csi_grid_sweep_weather_only", "baseline_mlp_single_csi"),
        "physmask": "baseline_mlp_dual_physmask_grid_sweep_physics_weight",
    },
    "LSTM": {
        "dual": ("baseline_lstm_dual_nomask_grid_sweep_weather_only", "baseline_lstm_dual_nomask"),
        "single": ("baseline_lstm_single_csi_grid_sweep_weather_only", "baseline_lstm_single_csi"),
        "physmask": "baseline_lstm_dual_physmask_grid_sweep_physics_weight",
    },
    "CVAE": {
        "dual": ("baseline_cvae_dual_nomask_grid_sweep_weather_only", "baseline_cvae_dual_nomask"),
        "single": ("baseline_cvae_single_csi_grid_sweep_weather_only", "baseline_cvae_single_csi"),
        "physmask": "baseline_cvae_dual_physmask_grid_sweep_physics_weight",
    },
    "KNN": {
        "dual": ("baseline_knn_dual_grid_sweep_weather_only", "baseline_knn_dual"),
        "single": ("baseline_knn_single_csi_grid_sweep_weather_only", "baseline_knn_single_csi"),
        "physmask": "baseline_knn_dual_physmask_grid_sweep_physics_weight",
    },
    "Linear": {
        "dual": ("baseline_linear_dual_grid_sweep_weather_only", "baseline_linear_dual"),
        "single": ("baseline_linear_single_csi_grid_sweep_weather_only", "baseline_linear_single_csi"),
        "physmask": "baseline_linear_dual_physmask_grid_sweep_physics_weight",
    },
    "SVR": {
        "dual": ("baseline_svr_dual_grid_sweep_weather_only", "baseline_svr_dual"),
        "single": ("baseline_svr_single_csi_grid_sweep_weather_only", "baseline_svr_single_csi"),
        "physmask": "baseline_svr_dual_physmask_grid_sweep_physics_weight",
    },
}

# --mode models --config <name>: one fixed (read_mode, weather) condition applied to every
# model in one figure -- never contrasted within a figure (that's what --mode physics's
# tiers, or separate --config figures, are for).
CONFIGS = {
    "dual_wx_off": ("dual", False),
    "dual_wx_on": ("dual", True),
    "single_wx_off": ("single", False),
    "single_wx_on": ("single", True),
}


def physmask_run(physmask_sweep: str, pw: str) -> str:
    """'baseline_mlp_dual_physmask_grid_sweep_physics_weight' -> 'baseline_mlp_dual_physmask-pw=0.0'"""
    prefix = physmask_sweep.removesuffix("_grid_sweep_physics_weight")
    return f"{prefix}-pw={pw}"


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
        "figure.dpi": 350,
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
        # Ground truth, not target-dependent -- always present in `true`
        # regardless of predict_targets. Needed for the physical export
        # floor (export = max(-net_demand, 0)), not just plain load/pv plots.
        "true_net": true[:, DisaggregationMetrics.TRUE_NET, :].numpy(),
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
        "true_net": ref["true_net"],
        "timestamps": ref["timestamps"],
        "user_ids": ref["user_ids"],
        "n_seeds": len(per_seed),
        # discover_manifest() spreads _parse_run_tag(run_name) into every row,
        # so a 'pw' column (string "0.0"/"0.3") already exists for physics
        # sweeps -- used by --mode violation to identify baseline vs.
        # physics-constrained runs without string-matching run_name.
        "pw": rows["pw"].iloc[0] if "pw" in rows.columns else None,
    }


def select_best_seed_run(rows: pd.DataFrame, target: str):
    """
    Like aggregate_run, but instead of averaging every seed, runs inference
    on only the single seed with the lowest test_loss (this project's
    checkpoint-selection metric throughout, cfg.metric_best: loss) and
    returns that seed's raw q50/qlo/qhi untouched.

    For publication figures we deliberately don't want a seed-averaged
    profile (it's not a real trajectory any model actually produced) --
    one real checkpoint's real prediction is what should be shown.
    """
    runnable = rows[rows["test_loss"].notna()].sort_values("test_loss")
    for _, row in runnable.iterrows():
        res = _run_predictions(row, target)
        if res is not None:
            res["n_seeds"] = 1
            res["seed"] = int(row["seed"])
            res["test_loss"] = float(row["test_loss"])
            res["pw"] = row.get("pw")
            return res
    return None


def load_named_run(manifest: pd.DataFrame, sweep: str, run_name: str, target: str, best_seed: bool = False):
    """
    Fetch exactly one (sweep, run_name)'s seeds from the full manifest and
    reduce them to one plottable series -- the cross-sweep counterpart to the
    __main__ block's old single-sweep `sub.groupby("run_name")` loop, needed
    once comparisons span multiple models (each living in its own sweep dir)
    rather than multiple configs within one model's sweep.

    best_seed=True picks the single lowest-test_loss seed (select_best_seed_run,
    for publication figures); False (default) averages every seed
    (aggregate_run, the original --mode mae/mode models matplotlib behavior).

    Returns None (with a printed reason) if the run isn't found/runnable/
    doesn't predict `target`, so a batch of models can skip one bad entry
    without aborting the whole comparison.
    """
    rows = manifest[(manifest["sweep"] == sweep) & (manifest["run_name"] == run_name)]
    rows = rows[rows["ckpt_path"].notna() | rows["model_type"].isin(CACHE_FIT_MODEL_TYPES)]
    if rows.empty:
        print(f"  skip {sweep}/{run_name}: no runnable rows found")
        return None
    res = select_best_seed_run(rows, target) if best_seed else aggregate_run(rows, target)
    if res is None:
        print(f"  skip {sweep}/{run_name}: doesn't predict {target!r}, or no checkpoints/dataset")
    return res


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


def rank_models_by_mae(manifest: pd.DataFrame, config: str, exclude: set[str] | None = None) -> list[str]:
    """
    Ranks MODEL_CONFIGS labels by test_mae_pv (best seed's, i.e. min across
    that run's seeds) for the given --mode models --config, ascending
    (best first). Deliberately test_mae_pv, not test_loss: test_loss isn't
    comparable across models here since CVAE trains against a VAE-ELBO loss
    (large-magnitude, can be negative) while every other model uses the
    shared masked pinball loss (small positive) -- test_mae_pv is denormalized
    physical-Watt MAE, valid to compare regardless of loss function.
    """
    exclude = exclude or set()
    read_mode, wx = CONFIGS[config]
    scored = []
    for label, sweeps in MODEL_CONFIGS.items():
        if label in exclude:
            continue
        sweep, run_prefix = sweeps[read_mode]
        run_name = f"{run_prefix}-wx={wx}"
        rows = manifest[(manifest["sweep"] == sweep) & (manifest["run_name"] == run_name)]
        rows = rows[rows["test_mae_pv"].notna()]
        if rows.empty:
            continue
        scored.append((label, rows["test_mae_pv"].min()))
    scored.sort(key=lambda t: t[1])
    return [label for label, _ in scored]


def find_window_start(timestamps: list, days: int) -> int:
    """First local-midnight-aligned index with `days` full days remaining."""
    steps = days * STEPS_PER_DAY
    for i, t in enumerate(timestamps):
        local = t.astimezone(AMSTERDAM)
        if local.hour == 0 and local.minute == 0 and i + steps <= len(timestamps):
            return i
    return 0  # no clean local-midnight boundary found — fall back to the start


def real_export_violation(true_net: np.ndarray, real: np.ndarray, mask: np.ndarray) -> dict:
    """
    Like DisaggregationMetrics.export_violation, but scored against the REAL
    (ground-truth) PV trajectory, not any model's prediction. A household
    exporting more than its real PV ever generated is a metering/measurement
    artifact in the data itself (physically it shouldn't happen at all) --
    this is a property of the dataset, identical for every model sharing the
    same test split, unlike a model-prediction violation (which depends on
    which model/tier happens to be used as "the reference").
    """
    export_true = np.clip(-true_net, 0, None)
    valid = mask & np.isfinite(real)
    shortfall = np.where(valid, export_true - real, -np.inf)
    violated = valid & (shortfall > 1e-6)
    count = int(violated.sum())
    magnitude = float(shortfall[violated].mean()) if count > 0 else 0.0
    return {"count": count, "mean_magnitude": magnitude}


def select_household_real_violation(ref_res: dict, rank: int = 1) -> str:
    """
    Picks a household ranked by how often/severely its REAL PV trajectory
    dips below the physical export floor -- a data property, not a model-
    prediction property, so this needs only ONE reference run (any model,
    any tier -- real/true_net/mask are ground truth, identical across every
    model on the same test split), not every model's predictions pooled
    together. Also means the resulting household+window can be computed
    once and reused for every model's figure without re-deriving it per
    model (previously this recomputed a *prediction*-based score per model,
    which was both wrong -- a "violation" should describe the data, not
    whichever model happens to be the reference -- and needlessly slow,
    since it required every model's inference to be run before any
    selection could happen at all).

    Score = (count, mean_magnitude), sorted frequency-first then by severity
    -- the figure normalizes PV by the household's own peak (see
    models_profile.typ/physics_profile.typ), so a single huge-magnitude-but-
    rare violation in raw Watts can get normalized down to something
    visually indistinguishable, while a household with frequent (even if
    individually smaller) violations is far more likely to show up clearly
    somewhere in the chosen window.
    `rank` (1-indexed) is an escape hatch if the top-ranked household's
    violations don't fit cleanly into a single --days window.
    """
    scores = {}
    for i, uid in enumerate(ref_res["user_ids"]):
        v = real_export_violation(ref_res["true_net"][i], ref_res["real"][i], ref_res["mask"][i])
        if v["count"] > 0:
            scores[uid] = (v["count"], v["mean_magnitude"])

    if not scores:
        raise ValueError("No household has any real export_violation in the ground-truth data.")

    ranked = sorted(scores, key=scores.get, reverse=True)
    if rank < 1 or rank > len(ranked):
        raise ValueError(f"Only {len(ranked)} violating households available; rank={rank} requested.")
    return ranked[rank - 1]


def find_violation_window_real(ref_res: dict, uid: str, days: int) -> int:
    """
    Like find_window_start, but scores every valid local-midnight-aligned
    window by real_export_violation (count, then severity, then earliest)
    scoped to that window's slice, instead of just taking the first valid
    one -- so the rendered window actually contains the phenomenon being
    shown. Uses only the real/ground-truth trajectory (see
    select_household_real_violation for why), so this window, once found,
    is valid for every model -- no per-model recomputation needed.
    """
    steps = days * STEPS_PER_DAY
    timestamps = ref_res["timestamps"]
    i = ref_res["user_ids"].index(uid)
    true_net, real, mask = ref_res["true_net"][i], ref_res["real"][i], ref_res["mask"][i]

    candidates = [
        idx for idx, t in enumerate(timestamps)
        if t.astimezone(AMSTERDAM).hour == 0
        and t.astimezone(AMSTERDAM).minute == 0
        and idx + steps <= len(timestamps)
    ]
    if not candidates:
        return find_window_start(timestamps, days)

    def score(start):
        v = real_export_violation(
            true_net[start:start + steps], real[start:start + steps], mask[start:start + steps]
        )
        return (v["count"], v["mean_magnitude"], -start)

    return max(candidates, key=score)


def iso_local_date(ts) -> str:
    """Local (Amsterdam) calendar date of a timestamp -- matches the boundary
    convention every window-selection function above already uses, so a
    date derived from one series is safe to feed back into
    find_window_by_date on another."""
    return ts.astimezone(AMSTERDAM).date().isoformat()


def find_window_by_date(timestamps: list, date_str: str, days: int) -> int:
    """
    Locate the local-midnight index for an explicit calendar date, rather
    than searching for one. Needed to reuse one fixed household+day across
    several independently-loaded series (e.g. every model's physics export,
    or every --config in models mode) -- cross-comparability means literally
    the same calendar day, not each series re-deriving its own "best" window
    independently (which need not agree).
    """
    steps = days * STEPS_PER_DAY
    for i, t in enumerate(timestamps):
        local = t.astimezone(AMSTERDAM)
        if local.date().isoformat() == date_str and local.hour == 0 and local.minute == 0:
            if i + steps > len(timestamps):
                raise ValueError(f"Window starting {date_str} runs past the end of available timestamps.")
            return i
    raise ValueError(f"No local-midnight timestamp found for date {date_str!r}.")


# --- 4. PLOTTING ---
def plot_household_trajectory(
    run_results: dict,
    uid: str,
    target: str,
    days: int,
    out_path: str,
    window_start: int | None = None,
    violation_overlay: list[str] | None = None,
    show_bands: bool = True,
    labels: dict | None = None,
    title: str | None = None,
) -> None:
    """
    run_results keys are display labels (e.g. model names for --mode models,
    or tier names for --mode physics) -- not necessarily raw run_names.

    violation_overlay: ordered list of run_results keys to annotate, only
    meaningful for target == "pv". When given, adds the physical export
    floor (dashed line + shaded impossible zone below it), a per-run
    violation-count annotation for the plotted window, and markers at the
    *first* listed run's specifically-violated timesteps (intended to be the
    least-constrained/no-physics tier, so the markers show what the physics
    terms are meant to fix). None (default, or an empty comparison) leaves
    the plot to just the trajectories.

    show_bands: qlo-qhi shaded band per run. Off by default recommended once
    more than ~3 runs are overlaid (--mode models, 7 models) -- overlapping
    bands stop being readable; on is fine for small comparisons (--mode
    physics, 3 tiers).

    labels: optional {run_results key -> legend label} override (e.g. to
    show "no mask/loss" instead of a raw run_name); defaults to the key.

    title: one-line figure title (e.g. the fixed config a --mode models
    figure holds constant, since that's otherwise only encoded in the
    filename) -- kept short, figure is print-sized (4x3in).
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = labels or {}
    if title:
        ax.set_title(title, fontsize=7)

    ref = next(iter(run_results.values()))
    timestamps = ref["timestamps"]
    steps = days * STEPS_PER_DAY
    start = window_start if window_start is not None else find_window_start(timestamps, days)
    end = min(start + steps, len(timestamps))
    window_ts = [t.astimezone(AMSTERDAM) for t in timestamps[start:end]]

    real_drawn = False
    for i, (key, res) in enumerate(sorted(run_results.items())):
        j = res["user_ids"].index(uid)
        mask = res["mask"][j, start:end]
        real = np.where(mask, res["real"][j, start:end], np.nan)
        q50 = res["q50"][j, start:end]
        qlo = res["qlo"][j, start:end]
        qhi = res["qhi"][j, start:end]

        if not real_drawn:
            ax.plot(window_ts, real, color="#222222", linewidth=0.9, label="Real", zorder=10)
            real_drawn = True

        color = COLOR_MAP[i % len(COLOR_MAP)]
        n_seeds = res["n_seeds"]
        label = f"{labels.get(key, key)} (n={n_seeds})"
        ax.plot(window_ts, q50, color=color, linewidth=0.6, label=label)
        if show_bands:
            ax.fill_between(window_ts, qlo, qhi, color=color, alpha=0.15, linewidth=0)

    if violation_overlay and target == "pv":
        j = ref["user_ids"].index(uid)
        true_net_window = ref["true_net"][j, start:end]
        export_true = np.clip(-true_net_window, 0, None)

        ax.plot(
            window_ts, export_true, color="#B23A3A", linewidth=0.6, linestyle="--",
            label="Export (physical PV floor)", zorder=9,
        )
        ax.fill_between(
            window_ts, 0, export_true, color="#B23A3A", alpha=0.08, linewidth=0, zorder=0,
            label="Physically impossible (PV < export)",
        )

        annotation_lines = []
        for k, key in enumerate(violation_overlay):
            res = run_results[key]
            j2 = res["user_ids"].index(uid)
            mask_w = res["mask"][j2, start:end]
            q50_w = res["q50"][j2, start:end]
            v = DisaggregationMetrics.export_violation(true_net_window, q50_w, mask_w)
            annotation_lines.append(
                f"{labels.get(key, key)}: {v['count']} violation(s), "
                f"mean {v['mean_magnitude']:.0f} W"
            )

            if k == 0 and v["count"] > 0:
                shortfall = np.clip(export_true - q50_w, 0, None)
                violated = shortfall > 1e-6
                ax.scatter(
                    np.array(window_ts)[violated], q50_w[violated],
                    color="#B23A3A", marker="x", s=14, zorder=11,
                    label=f"{labels.get(key, key)} violated timestep",
                )

        ax.text(
            0.02, 0.95, "\n".join(annotation_lines), transform=ax.transAxes,
            fontsize=6, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "linewidth": 0.4},
        )

    ax.set_ylabel(f"{target.upper()} (W)")
    fig.supxlabel("Time")
    fig.autofmt_xdate()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), ncols=3, frameon=False, fontsize=5.5)
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
        description="Overlay predicted-vs-real disaggregation trajectories."
    )
    parser.add_argument(
        "--mode",
        choices=["models", "physics", "mae"],
        default="models",
        help=(
            "'models' (default): every model in MODEL_CONFIGS, one fixed --config held "
            "constant across all of them, overlaid on one household/window -- the cross-model "
            "comparison figure. Never contrasts configs within a figure -- generate one figure "
            "per --config if you want more than one. "
            "'physics': one model (--model), three tiers (no mask/loss, mask only, mask+loss) "
            "on a household/window chosen for a bad no-mask/no-loss PV<export violation -- "
            "always one model per figure. "
            "'mae': legacy single-sweep mode (--sweep/--runs/--target), household ranked by "
            "pooled MAE -- kept for ad-hoc single-sweep debugging, not the recommended path."
        ),
    )
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default=None, help="--mode physics only")
    parser.add_argument("--models", default=None, help="--mode models only: comma-separated subset of MODEL_CONFIGS (default: all)")
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS),
        default="dual_wx_off",
        help="--mode models only: the one condition held fixed across all models in the figure",
    )
    parser.add_argument("--sweep", default=None, help="--mode mae only: sweep dir name under results/")
    parser.add_argument("--runs", default=None, help="--mode mae only: optional substring filter on run_name")
    parser.add_argument("--target", choices=["load", "pv"], default=None, help="--mode mae only (models/physics are PV-only)")
    parser.add_argument(
        "--household",
        default="auto:median",
        help=(
            "'auto:best' | 'auto:median' | 'auto:worst' | an explicit user_id (--mode models/mae); "
            "'auto:violation' | 'auto:violation:<rank>' (--mode physics, ignored otherwise)"
        ),
    )
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--out", default="trajectory.pdf")
    args = parser.parse_args()

    init_style()

    if args.mode == "models":
        manifest = discover_manifest()
        wanted = args.models.split(",") if args.models else sorted(MODEL_CONFIGS)
        unknown = set(wanted) - set(MODEL_CONFIGS)
        if unknown:
            raise SystemExit(f"Unknown model(s) in --models: {sorted(unknown)!r}; choices are {sorted(MODEL_CONFIGS)!r}")
        read_mode, wx = CONFIGS[args.config]

        run_results = {}
        for label in wanted:
            sweep, run_prefix = MODEL_CONFIGS[label][read_mode]
            run_name = f"{run_prefix}-wx={wx}"
            res = load_named_run(manifest, sweep, run_name, "pv")
            if res is not None:
                run_results[label] = res
        if not run_results:
            raise SystemExit(f"No requested model has a runnable checkpoint for --config {args.config!r}.")

        if args.household.startswith("auto:"):
            uid = select_household(run_results, args.household.split(":", 1)[1])
        else:
            uid = args.household

        config_title = f"{read_mode}-read, weather {'on' if wx else 'off'}"
        plot_household_trajectory(
            run_results, uid, "pv", args.days, args.out, show_bands=False, title=config_title,
        )

    elif args.mode == "physics":
        if args.model is None:
            raise SystemExit("--mode physics needs --model <name>; choices are " + repr(sorted(MODEL_CONFIGS)))
        manifest = discover_manifest()
        nomask_sweep, nomask_run_prefix = MODEL_CONFIGS[args.model]["dual"]
        nomask_run = f"{nomask_run_prefix}-wx=False"
        physmask_sweep = MODEL_CONFIGS[args.model]["physmask"]

        tier_labels = {
            "no_mask_no_loss": "no mask/loss",
            "mask_only": "mask only (pw=0.0)",
            "mask_and_loss": "mask+loss (pw=0.3)",
        }
        run_results = {
            "no_mask_no_loss": load_named_run(manifest, nomask_sweep, nomask_run, "pv"),
            "mask_only": load_named_run(manifest, physmask_sweep, physmask_run(physmask_sweep, "0.0"), "pv"),
            "mask_and_loss": load_named_run(manifest, physmask_sweep, physmask_run(physmask_sweep, "0.3"), "pv"),
        }
        missing = [k for k, v in run_results.items() if v is None]
        if missing:
            raise SystemExit(f"--mode physics for {args.model!r}: missing tier(s) {missing!r} (no runnable checkpoint).")

        if args.household.startswith("auto:violation"):
            parts = args.household.split(":")
            rank = int(parts[2]) if len(parts) > 2 else 1
            uid = select_household_real_violation(run_results["no_mask_no_loss"], rank)
        elif args.household.startswith("auto:"):
            uid = select_household(run_results, args.household.split(":", 1)[1])
        else:
            uid = args.household

        window_start = find_violation_window_real(run_results["no_mask_no_loss"], uid, args.days)
        plot_household_trajectory(
            run_results, uid, "pv", args.days, args.out,
            window_start=window_start,
            violation_overlay=["no_mask_no_loss", "mask_only", "mask_and_loss"],
            show_bands=True,
            labels=tier_labels,
            title=args.model,
        )

    else:  # legacy single-sweep mode
        if not args.sweep:
            raise SystemExit("--mode mae needs --sweep <dir>.")
        if args.target is None:
            raise SystemExit("--target is required when --mode mae.")

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
