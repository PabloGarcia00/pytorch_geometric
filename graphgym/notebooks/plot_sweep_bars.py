"""
Bar charts (mean +/- std over seeds) for the six weather/physics/read-mode
ablations, built on top of notebooks/metrics_long.csv (see calculate_metrics.py).

Usage:
    python notebooks/plot_sweep_bars.py
    python notebooks/plot_sweep_bars.py --sweeps sub_optimal_weather physics_pw1
    python notebooks/plot_sweep_bars.py --no-baselines
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_CSV = Path(__file__).resolve().parent / "metrics_long.csv"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent / "figures"

METRIC_LABELS = {
    "mae": "MAE",
    "r2": "R$^2$",
    "export_violation_rate": "PV<Export Rate",
    "export_violation_mag": "PV<Export Mag (W)",
}
TARGET_LABELS = {"load": "Load", "pv": "PV"}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

MAE_R2_PANELS = [("mae", "load"), ("mae", "pv"), ("r2", "load"), ("r2", "pv")]
PHYSICS_PANELS = MAE_R2_PANELS + [
    ("export_violation_rate", "pv"),
    ("export_violation_mag", "pv"),
]

# baselines that share the plain weather-only (wx=True/False, no coverage axis)
# sweep shape, matching the physics_* ablations
BASELINE_SWEEPS = {
    "CVAE": "baseline_cvae_grid_sweep_weather_only",
    "LSTM": "baseline_lstm_grid_sweep_weather_only",
    "MLP": "baseline_mlp_grid_sweep_weather_only",
}


def _parse_run_tag(run_name: str) -> dict:
    """'st_sgc_caps_single_full_optimal-wx=True' -> {'model_name': ..., 'wx': True}"""
    base, *params = run_name.split("-")
    tag: dict = {"model_name": base}
    for param in params:
        if "=" not in param:
            continue
        key, value = param.split("=", 1)
        tag[key] = value if value not in ("True", "False") else value == "True"
    return tag


def load_long(csv_path: Path = METRICS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    tags = df["run_name"].apply(_parse_run_tag).apply(pd.Series)
    return pd.concat([df, tags], axis=1)


def _last_stats(path: Path) -> dict:
    """stats.json is JSON-lines (one record per epoch); return the last one."""
    if not path.exists():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def best_seed_per_run(results_root: Path, sweeps: set) -> dict:
    """(sweep, run_name) -> seed with the lowest val loss, straight from
    results/<sweep>/<run_name>/<seed>/val/stats.json — no torch/checkpoint needed,
    so this works even while sweeps are still training."""
    best = {}
    for sweep in sweeps:
        sweep_dir = results_root / sweep
        if not sweep_dir.is_dir():
            continue
        for run_dir in sorted(sweep_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            candidates = []
            for seed_dir in run_dir.iterdir():
                if not (seed_dir.is_dir() and seed_dir.name.isdigit()):
                    continue
                loss = _last_stats(seed_dir / "val" / "stats.json").get("loss")
                if loss is not None:
                    candidates.append((loss, int(seed_dir.name)))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                best[(sweep, run_dir.name)] = candidates[0][1]
    return best


def restrict_to_best_seed(df: pd.DataFrame, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Keep only the lowest-val-loss seed per (sweep, run_name). Sweeps aren't done
    seeding today, so this reports one point estimate per arm instead of a mean over
    an incomplete/uneven seed count."""
    best = best_seed_per_run(results_root, set(df["sweep"].unique()))
    if not best:
        return df
    best_df = pd.DataFrame(
        [(s, r, seed) for (s, r), seed in best.items()], columns=["sweep", "run_name", "best_seed"]
    )
    merged = df.merge(best_df, on=["sweep", "run_name"], how="left")
    has_best = merged["best_seed"].notna()
    keep = has_best & (merged["seed"] == merged["best_seed"])
    dropped_runs = sorted(set(map(tuple, merged.loc[~has_best, ["sweep", "run_name"]].values.tolist())))
    if dropped_runs:
        print(f"    ! no val/stats.json for: {dropped_runs} — dropped (can't pick a best seed)")
    return merged[keep].drop(columns=["best_seed"])


def _warn_missing(stats, index_labels, metric, target):
    missing = stats["mean"].isna()
    if missing.any():
        print(f"    ! no data yet for {metric}/{target}: {[index_labels[i] for i in stats.index[missing]]}")


def _bar_panel(ax, sub, metric, target, category_col, category_order, category_labels):
    rows = sub[(sub["metric"] == metric) & (sub["target"] == target)]
    stats = rows.groupby(category_col)["value"].agg(["mean", "std", "count"]).reindex(category_order)
    _warn_missing(stats, category_labels, metric, target)

    x = range(len(category_order))
    ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0), capsize=4, color=PALETTE[: len(category_order)])
    ax.set_xticks(list(x))
    ax.set_xticklabels([category_labels[c] for c in category_order], fontsize=7)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {TARGET_LABELS.get(target, target)}", fontsize=8)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.axhline(0, color="black", linewidth=0.6)


def _grouped_bar_panel(ax, sub, metric, target, group_col, group_order, group_labels, hue_col, hue_order, hue_labels):
    rows = sub[(sub["metric"] == metric) & (sub["target"] == target)]
    stats = rows.groupby([group_col, hue_col])["value"].agg(["mean", "std", "count"])

    n_hues = len(hue_order)
    width = 0.8 / n_hues
    x_base = np.arange(len(group_order))

    for i, hue in enumerate(hue_order):
        means, stds = [], []
        for g in group_order:
            if (g, hue) in stats.index:
                row = stats.loc[(g, hue)]
                means.append(row["mean"])
                stds.append(0.0 if pd.isna(row["std"]) else row["std"])
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (i - (n_hues - 1) / 2) * width
        ax.bar(
            x_base + offset, means, width=width, yerr=stds, capsize=3,
            label=hue_labels[hue], color=PALETTE[i % len(PALETTE)],
        )

    flat_labels = {(g, h): f"{group_labels[g]}/{hue_labels[h]}" for g in group_order for h in hue_order}
    _warn_missing(
        stats.reindex(pd.MultiIndex.from_product([group_order, hue_order])), flat_labels, metric, target
    )

    ax.set_xticks(list(x_base))
    ax.set_xticklabels([group_labels[g] for g in group_order], fontsize=7)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {TARGET_LABELS.get(target, target)}", fontsize=8)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=5.5)


PANEL_W, PANEL_H, DPI = 4, 3, 350
CAPTION = "bars = single best seed by val loss per arm (sweeps still in progress, no error bars)"


def _finish_figure(fig, axes, panels, name, title, nrows):
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig_h = PANEL_H * nrows
    # reserve a fixed header (title + caption) in inches so it scales with nrows
    # instead of a fixed figure-fraction, which breaks for short (2-row) figures
    title_y = 1 - (0.28 / fig_h)
    caption_y = 1 - (0.5 / fig_h)
    body_frac = 1 - (0.68 / fig_h)
    fig.suptitle(title, fontsize=9, y=title_y, wrap=True)
    fig.text(0.5, caption_y, CAPTION, ha="center", fontsize=5.5, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, body_frac])
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"    -> {out_path}")
    return out_path


def plot_sweep(sub, name, title, panels, category_col, category_order, category_labels, ncols=2):
    print(f"[{name}] {title}  (n_rows={len(sub)})")
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrows), squeeze=False)
    for ax, (metric, target) in zip(axes.flat, panels):
        _bar_panel(ax, sub, metric, target, category_col, category_order, category_labels)
    return _finish_figure(fig, axes, panels, name, title, nrows)


def plot_grouped_sweep(sub, name, title, panels, group_col, group_order, group_labels, hue_col, hue_order, hue_labels, ncols=2):
    print(f"[{name}] {title}  (n_rows={len(sub)})")
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrows), squeeze=False)
    for ax, (metric, target) in zip(axes.flat, panels):
        _grouped_bar_panel(ax, sub, metric, target, group_col, group_order, group_labels, hue_col, hue_order, hue_labels)
    return _finish_figure(fig, axes, panels, name, title, nrows)


# ── sweep definitions ───────────────────────────────────────────────────────
WX_CATEGORY_ORDER = [False, True]
WX_CATEGORY_LABELS = {False: "No Weather", True: "Weather"}

SIMPLE_SWEEPS = {
    "sub_optimal_weather": {
        "title": "Weather vs No-Weather — Sub-optimal Coverage",
        "sweep": "st_sgc_caps_single_full_suboptimal_grid_sweep_weather_only",
        "panels": MAE_R2_PANELS,
    },
    "optimal_weather": {
        "title": "Weather vs No-Weather — Optimal Coverage",
        "sweep": "st_sgc_caps_single_full_optimal_grid_sweep_weather_only",
        "panels": MAE_R2_PANELS,
    },
}

PHYSICS_SWEEPS = {
    "physics_noloss": {
        "title": "Physics Mask, No Physics Loss (weight=0) — vs Baselines",
        "sweep": "earne_phys_single_noloss_grid_sweep_weather_only",
        "primary_label": "EARNE (phys)",
        "panels": PHYSICS_PANELS,
    },
    "physics_pw03": {
        "title": "Physics Mask + Physics Loss (weight=0.3) — vs Baselines",
        "sweep": "earne_phys_single_pw03_grid_sweep_weather_only",
        "primary_label": "EARNE (phys)",
        "panels": PHYSICS_PANELS,
    },
    "physics_pw1": {
        "title": "Physics Mask + Physics Loss (weight=1.0) — vs Baselines",
        "sweep": "earne_phys_single_grid_sweep_weather_only",
        "primary_label": "EARNE (phys)",
        "panels": PHYSICS_PANELS,
    },
}

DUAL_VS_SINGLE = {
    "title": "Dual-Read vs Single-Read × Weather — Optimal Coverage (ST-SGCCaps)",
    "sweeps": {
        "single": "st_sgc_caps_single_full_optimal_grid_sweep_weather_only",
        "dual": "st_sgc_caps_dual_full_optimal_grid_sweep_weather_only",
    },
    "panels": MAE_R2_PANELS,
    "row_filter": None,
}

# ST-SGCCaps single-read isn't trained yet, so use EARNE-learned-corr (which has
# both read types fully done) for the single-vs-dual x weather comparison,
# restricted to optimal coverage (span=True) to match the original request.
DUAL_VS_SINGLE_WEATHER = {
    "title": "Dual vs Single Read × Weather (EARNE learned-corr, optimal span)",
    "sweeps": {
        "single": "earne_learned_corr_single_grid_sweep_coverage_weather",
        "dual": "earne_learned_corr_dual_grid_sweep_coverage_weather",
    },
    "panels": MAE_R2_PANELS,
    "row_filter": lambda d: d[d["span"] == True],  # noqa: E712
}

RW_CATEGORY_ORDER = ["single_nowx", "single_wx", "dual_nowx", "dual_wx"]
RW_CATEGORY_LABELS = {
    "single_nowx": "Single\nNo-Weather",
    "single_wx": "Single\nWeather",
    "dual_nowx": "Dual\nNo-Weather",
    "dual_wx": "Dual\nWeather",
}


def build_simple_sub(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    return df[df["sweep"] == sweep].copy()


def build_read_weather_sub(df: pd.DataFrame, sweeps: dict, row_filter=None) -> pd.DataFrame:
    frames = []
    for read_type, sweep in sweeps.items():
        sub = df[df["sweep"] == sweep].copy()
        if row_filter is not None:
            sub = row_filter(sub)
        sub["read_type"] = read_type
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)
    combined["category"] = combined["read_type"] + "_" + combined["wx"].map({True: "wx", False: "nowx"})
    return combined


def complete_hues(sub: pd.DataFrame, hue_col: str, hue_order: list, group_col: str, group_order: list) -> list:
    """Baselines are all-or-nothing: only keep a hue if it has at least one row for
    every value in group_order (e.g. both wx=True and wx=False)."""
    keep = []
    for hue in hue_order:
        present = set(sub.loc[sub[hue_col] == hue, group_col].unique())
        if set(group_order).issubset(present):
            keep.append(hue)
    return keep


def build_model_weather_sub(df: pd.DataFrame, primary_label: str, primary_sweep: str, baselines: dict) -> pd.DataFrame:
    frames = []
    primary = df[df["sweep"] == primary_sweep].copy()
    primary["model"] = primary_label
    frames.append(primary)
    for label, sweep in baselines.items():
        sub = df[df["sweep"] == sweep].copy()
        sub["model"] = label
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=METRICS_CSV)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        # st_sgc_caps-backed sweeps (sub_optimal_weather, optimal_weather,
        # dual_vs_single_optimal) are still training and omitted by default —
        # pass them explicitly via --sweeps once those runs checkpoint.
        default=list(PHYSICS_SWEEPS) + ["dual_vs_single_weather"],
        help="Subset of sweep names to plot",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Plot physics_* sweeps as a single model, without the CVAE/LSTM/MLP baseline bars",
    )
    args = parser.parse_args()

    df = restrict_to_best_seed(load_long(args.csv))
    available = set(df["sweep"].unique())
    # bars = best-seed-by-val-loss point estimate; sweeps are still filling in,
    # so we build every chart with whatever arms exist rather than skipping.

    for name in args.sweeps:
        if name in ("dual_vs_single_optimal", "dual_vs_single_weather"):
            spec = DUAL_VS_SINGLE if name == "dual_vs_single_optimal" else DUAL_VS_SINGLE_WEATHER
            missing = [s for s in spec["sweeps"].values() if s not in available]
            if missing:
                print(f"[{name}] missing sweep(s), plotting with gaps: {missing}")
            if len(missing) == len(spec["sweeps"]):
                print(f"[{name}] SKIPPED — no data for either side yet")
                continue
            sub = build_read_weather_sub(df, spec["sweeps"], row_filter=spec["row_filter"])
            plot_sweep(sub, name, spec["title"], spec["panels"], "category", RW_CATEGORY_ORDER, RW_CATEGORY_LABELS)

        elif name in PHYSICS_SWEEPS:
            spec = PHYSICS_SWEEPS[name]
            if spec["sweep"] not in available:
                print(f"[{name}] primary sweep not ready yet ({spec['sweep']}) — plotting baselines only")
            if args.no_baselines:
                sub = build_simple_sub(df, spec["sweep"])
                if sub.empty:
                    print(f"[{name}] SKIPPED — no data at all yet")
                    continue
                plot_sweep(sub, name, spec["title"], spec["panels"], "wx", WX_CATEGORY_ORDER, WX_CATEGORY_LABELS)
                continue
            candidate_baselines = {label: s for label, s in BASELINE_SWEEPS.items() if s in available}
            sub = build_model_weather_sub(df, spec["primary_label"], spec["sweep"], candidate_baselines)
            if sub.empty:
                print(f"[{name}] SKIPPED — no data at all yet")
                continue
            complete_baselines = complete_hues(sub, "model", list(candidate_baselines), "wx", WX_CATEGORY_ORDER)
            omitted = [b for b in candidate_baselines if b not in complete_baselines]
            if omitted:
                print(f"    ! baselines incomplete for {WX_CATEGORY_ORDER}, omitting entirely: {omitted}")
            hue_order = [spec["primary_label"]] + complete_baselines
            hue_labels = {h: h for h in hue_order}
            plot_grouped_sweep(
                sub, name, spec["title"], spec["panels"],
                "wx", WX_CATEGORY_ORDER, WX_CATEGORY_LABELS,
                "model", hue_order, hue_labels,
            )

        else:
            spec = SIMPLE_SWEEPS[name]
            if spec["sweep"] not in available:
                print(f"[{name}] SKIPPED — no data at all yet: {spec['sweep']}")
                continue
            sub = build_simple_sub(df, spec["sweep"])
            plot_sweep(sub, name, spec["title"], spec["panels"], "wx", WX_CATEGORY_ORDER, WX_CATEGORY_LABELS)


if __name__ == "__main__":
    main()
