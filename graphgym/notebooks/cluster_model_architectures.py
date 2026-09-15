"""
cluster_model_architectures.py

Hierarchical (Ward, correlation-distance) clustering of the 7 disaggregation
model architectures (CVAE, KNN, Linear, SVR, LSTM, MLP, ST-GNN) by how
similarly they *react* to pipeline ablations -- not by their absolute
accuracy level. Reads the already-computed notebooks/metrics_long.csv (no
custom_graphgym/torch_geometric dependency).

Feature vector, per architecture: %-change-from-baseline RMSE (on pv) across
10 fixed ablation cells (dual-read, dual+weather, single+weather, resolution
5/30/60min, cold-start coverage, physics mask pw=0.0/0.3, physics-loss-only
pw=0.3). Clustering on the *relative* reaction (not raw RMSE) is the point:
two models with very different absolute accuracy can still cluster together
if they respond to an ablation the same way, and two similarly-accurate
models can split apart if one degrades sharply under an ablation the other
shrugs off.

Two baselines, not one, to avoid confounding effects:
  - BASE_SINGLE: single-read, no weather, no mask, pw=0.0, native res --
    used for cells 1-7 (read-mode, weather, resolution, coverage).
  - BASE_DUAL: dual-read, no weather, no mask -- used for cells 8-10
    (physics). Every physics ablation in this dataset only ever exists in a
    dual-read context (see visualization/trajectory.py's MODEL_CONFIGS
    comment: "no single-read physics experiment exists"), so comparing a
    physics cell to BASE_SINGLE would silently conflate "physics effect"
    with "dual-read effect". Cell 1 (dual_read) separately captures the
    dual-read effect on its own, decomposing what would otherwise be one
    muddled comparison into two clean ones.

KNN caveat: KNN (custom_graphgym/train/knn_train.py, max_epoch=0) never
trains -- it's a non-gradient lookup model, so `physics_weight` cannot
affect it structurally. If KNN clusters with CVAE (whose stability instead
comes from an NLL-anchored loss that the physics term can't dominate), that
is two different reasons for looking similar, not evidence KNN "learned" to
ignore the physics loss. KNN is also missing 2 of the 10 cells entirely
(coldstart_coverage, resolution_5min -- 0 rows in metrics_long.csv, not a
bug) and has seed=0-only data for several others; the correlation matrix is
computed pairwise-complete by default so these gaps don't force every other
architecture to lose those cells too (see --strict-complete-cells to
compare against the alternative).

coverage_eval caveat: `coverage_eval_*_suboptimal` in metrics_long.csv is
currently a SUPERSET of `_optimal` (112 households vs. 62, not the disjoint
50 cold-start-only households from held_out_metrics_*.csv) -- this script
uses that data as-is. The "cold-start" framing of cell 7 should be read
against this superset semantics, not a disjoint cold-start population.

Ward-on-correlation-distance caveat: Ward's Lance-Williams update is only
rigorously "minimum variance" under true (squared-)Euclidean distance;
correlation distance isn't that. The cophenetic correlation coefficient is
printed as a diagnostic so the dendrogram's merge heights are read as a
useful heuristic, not a statistically pure quantity.

Usage:
    python notebooks/cluster_model_architectures.py
    python notebooks/cluster_model_architectures.py --metric nrmse
    python notebooks/cluster_model_architectures.py --strict-complete-cells
    python notebooks/cluster_model_architectures.py --include-tstr
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform

METRICS_CSV = Path(__file__).resolve().parent / "metrics_long.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
FEATURES_CSV = (
    Path(__file__).resolve().parent / "model_arch_degradation_features.csv"
)

# Same (model_type, node_encoder_name, graph_mode) -> architecture mapping
# validated against metrics_long.csv throughout this project's ablation
# analysis.
ARCH_MAP = {
    ("baseline_cvae", "earne_temporal", "spatial_knn"): "CVAE",
    ("baseline_knn", "earne_temporal", "learned_corr"): "KNN",
    ("baseline_linear", "earne_temporal", "learned_corr"): "Linear",
    ("baseline_svr", "earne_temporal", "learned_corr"): "SVR",
    ("earne_network", "baseline_lstm_temporal", "spatial_knn"): "LSTM",
    ("earne_network", "baseline_mlp_temporal", "spatial_knn"): "MLP",
    ("earne_network", "earne_temporal", "learned_corr"): "GNN",
}
ARCH_ORDER = ["CVAE", "KNN", "Linear", "SVR", "LSTM", "MLP", "GNN"]
# Display-only rename, matching the environment-wide convention already
# established in visualization/trajectory.py (MODEL_CONFIGS uses "ST-GNN").
ARCH_LABEL = {a: a for a in ARCH_ORDER}
ARCH_LABEL["GNN"] = "ST-GNN"

# Short keys used in res_eval_*/coverage_eval_* run names.
ARCH_KEY = {
    "CVAE": "cvae",
    "KNN": "knn",
    "Linear": "linear",
    "SVR": "svr",
    "LSTM": "lstm",
    "MLP": "mlp",
    "GNN": "gnn",
}

# (sweep, run_name) for the single-read/no-weather/no-mask/pw=0.0 default,
# per architecture -- copied from visualization/trajectory.py's
# MODEL_CONFIGS["single"], independently verified against metrics_long.csv.
SINGLE_WEATHER = {
    "CVAE": (
        "baseline_cvae_single_csi_grid_sweep_weather_only",
        "baseline_cvae_single_csi",
    ),
    "KNN": (
        "baseline_knn_single_csi_grid_sweep_weather_only",
        "baseline_knn_single_csi",
    ),
    "Linear": (
        "baseline_linear_single_csi_grid_sweep_weather_only",
        "baseline_linear_single_csi",
    ),
    "SVR": (
        "baseline_svr_single_csi_grid_sweep_weather_only",
        "baseline_svr_single_csi",
    ),
    "LSTM": (
        "baseline_lstm_single_csi_grid_sweep_weather_only",
        "baseline_lstm_single_csi",
    ),
    "MLP": (
        "baseline_mlp_single_csi_grid_sweep_weather_only",
        "baseline_mlp_single_csi",
    ),
    "GNN": (
        "earne_learned_corr_single_csi_grid_sweep_weather_only",
        "earne_learned_corr_single_csi",
    ),
}

# (sweep, run_name) for the dual-read/no-weather/no-mask default, per
# architecture -- copied from MODEL_CONFIGS["dual"].
DUAL_WEATHER = {
    "CVAE": (
        "baseline_cvae_dual_nomask_grid_sweep_weather_only",
        "baseline_cvae_dual_nomask",
    ),
    "KNN": (
        "baseline_knn_dual_grid_sweep_weather_only",
        "baseline_knn_dual",
    ),
    "Linear": (
        "baseline_linear_dual_grid_sweep_weather_only",
        "baseline_linear_dual",
    ),
    "SVR": (
        "baseline_svr_dual_grid_sweep_weather_only",
        "baseline_svr_dual",
    ),
    "LSTM": (
        "baseline_lstm_dual_nomask_grid_sweep_weather_only",
        "baseline_lstm_dual_nomask",
    ),
    "MLP": (
        "baseline_mlp_dual_nomask_grid_sweep_weather_only",
        "baseline_mlp_dual_nomask",
    ),
    "GNN": (
        "earne_learned_corr_dual_grid_sweep_weather_only",
        "earne_learned_corr_dual",
    ),
}

# physmask sweep dir per architecture -- copied from MODEL_CONFIGS["physmask"].
PHYSMASK_SWEEP = {
    "CVAE": "baseline_cvae_dual_physmask_grid_sweep_physics_weight",
    "KNN": "baseline_knn_dual_physmask_grid_sweep_physics_weight",
    "Linear": "baseline_linear_dual_physmask_grid_sweep_physics_weight",
    "SVR": "baseline_svr_dual_physmask_grid_sweep_physics_weight",
    "LSTM": "baseline_lstm_dual_physmask_grid_sweep_physics_weight",
    "MLP": "baseline_mlp_dual_physmask_grid_sweep_physics_weight",
    "GNN": "earne_learned_corr_dual_physmask_grid_sweep_physics_weight",
}

# physics_weight_nomask sweep's run_name (mask=False, pw=0.3), per
# architecture -- verified against metrics_long.csv's physics_weight_nomask
# rows.
PW03_NOMASK_RUN = {
    "CVAE": "baseline_cvae_dual_nomask_pw03",
    "KNN": "baseline_knn_dual_pw03",
    "Linear": "baseline_linear_dual_pw03",
    "SVR": "baseline_svr_dual_pw03",
    "LSTM": "baseline_lstm_dual_nomask_pw03",
    "MLP": "baseline_mlp_dual_nomask_pw03",
    "GNN": "earne_learned_corr_dual_pw03",
}


def physmask_run(physmask_sweep: str, pw: str) -> str:
    """'baseline_mlp_dual_physmask_grid_sweep_physics_weight' ->
    'baseline_mlp_dual_physmask-pw=0.0'. Verbatim from
    visualization/trajectory.py."""
    prefix = physmask_sweep.removesuffix("_grid_sweep_physics_weight")
    return f"{prefix}-pw={pw}"


def load_long(csv_path: Path = METRICS_CSV) -> pd.DataFrame:
    """Load metrics_long.csv and derive the `arch` column via ARCH_MAP."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["arch"] = df.apply(
        lambda r: ARCH_MAP.get(
            (r["model_type"], r["node_encoder_name"], r["graph_mode"])
        ),
        axis=1,
    )
    return df


def _mean_value(
    df: pd.DataFrame,
    sweep: str,
    run_name: str,
    metric: str,
    arch: str,
    target: str = "pv",
) -> tuple[float, int]:
    """Mean of `value` over available seeds for one (arch, sweep, run_name,
    metric, target). Returns (nan, 0) if no matching rows exist."""
    sub = df[
        (df["arch"] == arch)
        & (df["sweep"] == sweep)
        & (df["run_name"] == run_name)
        & (df["metric"] == metric)
        & (df["target"] == target)
    ]
    if sub.empty:
        return float("nan"), 0
    return float(sub["value"].mean()), sub["seed"].nunique()


def _pct_change(ablation: float, baseline: float) -> float:
    """(ablation - baseline) / baseline * 100, nan-safe."""
    if pd.isna(ablation) or pd.isna(baseline) or baseline == 0:
        return float("nan")
    return (ablation - baseline) / baseline * 100.0


def build_cell_specs(include_tstr: bool = False) -> list[dict]:
    """Return the ablation-cell specs. Each spec resolves, per architecture,
    an ablation (sweep, run_name) and a baseline (sweep, run_name) -- both
    keyed by ARCH_KEY/SINGLE_WEATHER/DUAL_WEATHER/PHYSMASK_SWEEP/
    PW03_NOMASK_RUN so build_feature_matrix() can look them up uniformly."""

    def base_single(arch):
        return _weather_off_run(SINGLE_WEATHER[arch])

    def base_dual(arch):
        return _weather_off_run(DUAL_WEATHER[arch])

    specs = [
        {
            "name": "dual_read",
            "ablation": lambda a: _weather_off_run(DUAL_WEATHER[a]),
            "baseline": base_single,
        },
        {
            "name": "dual_weather",
            "ablation": lambda a: _weather_on_run(DUAL_WEATHER[a]),
            "baseline": base_single,
        },
        {
            "name": "weather_single",
            "ablation": lambda a: _weather_on_run(SINGLE_WEATHER[a]),
            "baseline": base_single,
        },
        {
            "name": "resolution_5min",
            "ablation": lambda a: (
                "resolution_eval",
                f"res_eval_{ARCH_KEY[a]}_5min",
            ),
            "baseline": base_single,
        },
        {
            "name": "resolution_30min",
            "ablation": lambda a: (
                "resolution_eval",
                f"res_eval_{ARCH_KEY[a]}_30min",
            ),
            "baseline": base_single,
        },
        {
            "name": "resolution_60min",
            "ablation": lambda a: (
                "resolution_eval",
                f"res_eval_{ARCH_KEY[a]}_60min",
            ),
            "baseline": base_single,
        },
        {
            "name": "coldstart_coverage",
            "ablation": lambda a: (
                "coverage_eval",
                f"coverage_eval_{ARCH_KEY[a]}_suboptimal",
            ),
            # Within-block baseline, NOT BASE_SINGLE: coverage_eval's
            # household population differs from the global default's, so
            # comparing suboptimal to BASE_SINGLE would confound "cold
            # start" with "which households are being scored".
            "baseline": lambda a: (
                "coverage_eval",
                f"coverage_eval_{ARCH_KEY[a]}_optimal",
            ),
        },
        {
            "name": "physics_mask_pw00",
            "ablation": lambda a: (
                PHYSMASK_SWEEP[a],
                physmask_run(PHYSMASK_SWEEP[a], "0.0"),
            ),
            "baseline": base_dual,
        },
        {
            "name": "physics_mask_pw03",
            "ablation": lambda a: (
                PHYSMASK_SWEEP[a],
                physmask_run(PHYSMASK_SWEEP[a], "0.3"),
            ),
            "baseline": base_dual,
        },
        {
            "name": "physics_noloss_pw03",
            "ablation": lambda a: (
                "physics_weight_nomask",
                PW03_NOMASK_RUN[a],
            ),
            "baseline": base_dual,
        },
    ]

    if include_tstr:
        # trtr is real-data-only training; trts_*/tstr_* substitute
        # synthetic data at train and/or test time. Uses trtr as its own
        # within-block baseline (different axis from the pipeline
        # ablations above -- comparing to BASE_SINGLE/BASE_DUAL would be
        # meaningless since tstr_trts_eval runs aren't organized by
        # read-mode/weather at all).
        for source in ("energydiff", "faraday", "guide_vae", "guide_vae_base"):
            specs.append(
                {
                    "name": f"trts_{source}",
                    "ablation": lambda a, s=source: (
                        "tstr_trts_eval",
                        f"trts_{s}",
                    ),
                    "baseline": lambda a: ("tstr_trts_eval", "trtr"),
                }
            )
            specs.append(
                {
                    "name": f"tstr_{source}",
                    "ablation": lambda a, s=source: (
                        "tstr_trts_eval",
                        f"tstr_{s}",
                    ),
                    "baseline": lambda a: ("tstr_trts_eval", "trtr"),
                }
            )

    return specs


def _weather_on_run(sweep_run: tuple[str, str]) -> tuple[str, str]:
    """('baseline_mlp_dual_nomask_grid_sweep_weather_only',
    'baseline_mlp_dual_nomask') ->
    (..., 'baseline_mlp_dual_nomask-wx=True'). Both wx=False and wx=True
    variants carry the suffix -- there is no bare-prefix row (verified
    against metrics_long.csv's actual run_name values)."""
    sweep, run_prefix = sweep_run
    return sweep, f"{run_prefix}-wx=True"


def _weather_off_run(sweep_run: tuple[str, str]) -> tuple[str, str]:
    """Same as _weather_on_run but for the wx=False variant -- the default
    baseline condition."""
    sweep, run_prefix = sweep_run
    return sweep, f"{run_prefix}-wx=False"


def build_feature_matrix(
    df: pd.DataFrame, metric: str = "rmse", include_tstr: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features [7 archs x N cells] of %-change-from-baseline,
    seed_counts [debug frame, same shape, ablation-side seed count]).
    Prints a warning per (arch, cell) with 0 seeds (missing) and a note for
    <3 seeds."""
    specs = build_cell_specs(include_tstr=include_tstr)
    features = pd.DataFrame(
        index=ARCH_ORDER, columns=[s["name"] for s in specs], dtype=float
    )
    seed_counts = pd.DataFrame(
        index=ARCH_ORDER, columns=[s["name"] for s in specs], dtype=float
    )

    print(
        "NOTE: coverage_eval's 'suboptimal' run in metrics_long.csv is "
        "currently a SUPERSET of 'optimal' (112 vs 62 households), not the "
        "disjoint cold-start population from held_out_metrics_*.csv. The "
        "'coldstart_coverage' feature cell reflects that superset "
        "semantics.\n"
    )

    for spec in specs:
        for arch in ARCH_ORDER:
            a_sweep, a_run = spec["ablation"](arch)
            b_sweep, b_run = spec["baseline"](arch)
            a_val, a_n = _mean_value(df, a_sweep, a_run, metric, arch)
            b_val, b_n = _mean_value(df, b_sweep, b_run, metric, arch)
            features.loc[arch, spec["name"]] = _pct_change(a_val, b_val)
            seed_counts.loc[arch, spec["name"]] = a_n

            if a_n == 0:
                print(
                    f"  [missing] {arch} / {spec['name']}: no rows for "
                    f"({a_sweep}, {a_run})"
                )
            elif a_n < 3:
                print(
                    f"  [reduced-seed] {arch} / {spec['name']}: only "
                    f"{a_n} seed(s) for ({a_sweep}, {a_run})"
                )
            if b_n == 0:
                print(
                    f"  [missing baseline] {arch} / {spec['name']}: no "
                    f"rows for baseline ({b_sweep}, {b_run})"
                )

    return features, seed_counts


def correlation_distance_matrix(
    features: pd.DataFrame, strict_complete: bool = False
) -> pd.DataFrame:
    """Pairwise-complete (default) or drop-any-NaN-column (strict_complete)
    1 - Pearson r. Symmetrizes, zero-diagonal, clips negative floating-point
    noise. Prints per-pair N cells used."""
    feat = features.copy()
    if strict_complete:
        before = feat.shape[1]
        feat = feat.dropna(axis=1, how="any")
        dropped = before - feat.shape[1]
        print(
            f"\n--strict-complete-cells: dropped {dropped} of {before} "
            f"cells that had any missing architecture; {feat.shape[1]} "
            f"cells remain for every architecture.\n"
        )

    corr = feat.T.corr()  # pandas: pairwise-complete by default
    dist = 1.0 - corr
    dist = (dist + dist.T) / 2.0
    dist_vals = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_vals, 0.0)
    dist = pd.DataFrame(dist_vals, index=dist.index, columns=dist.columns)
    dist = dist.clip(lower=0.0)

    print("Pairwise cell counts used for correlation:")
    for i, a in enumerate(feat.index):
        for b in feat.index[i + 1 :]:
            n_cells = feat.loc[[a, b]].dropna(axis=1, how="any").shape[1]
            print(f"  {a} vs {b}: {n_cells}/{feat.shape[1]} cells")

    return dist


def run_ward(dist: pd.DataFrame) -> np.ndarray:
    """squareform(dist) -> linkage(..., method='ward'); prints the
    cophenetic correlation coefficient as a fit diagnostic (Ward assumes
    Euclidean input; correlation distance isn't strictly that)."""
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="ward")
    coph_corr, _ = cophenet(Z, condensed)
    print(
        f"\nCophenetic correlation coefficient: {coph_corr:.4f} "
        "(how faithfully the dendrogram preserves the pairwise distances; "
        "Ward isn't a pure fit for non-Euclidean correlation distance, so "
        "treat this as a heuristic diagnostic, not a guarantee)."
    )
    return Z


def init_style():
    """Matches visualization/trajectory.py's init_style()."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
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
        }
    )


def plot_dendrogram(Z: np.ndarray, labels: list[str], out_path: Path, title: str):
    init_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    dendrogram(
        Z,
        labels=[ARCH_LABEL.get(lbl, lbl) for lbl in labels],
        ax=ax,
        color_threshold=0,
        above_threshold_color="#34495e",
        leaf_font_size=9,
    )
    ax.set_ylabel("Ward distance (1 − Pearson r of %-degradation)")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved dendrogram to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=METRICS_CSV)
    parser.add_argument(
        "--metric",
        default="rmse",
        help="metrics_long.csv `metric` value to build feature cells from "
        "(default: rmse)",
    )
    parser.add_argument("--target", default="pv")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "model_arch_dendrogram.png",
    )
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument(
        "--strict-complete-cells",
        action="store_true",
        help="drop any ablation cell missing for any architecture, "
        "instead of the default pairwise-complete correlation",
    )
    parser.add_argument(
        "--include-tstr",
        action="store_true",
        help="also include tstr_trts_eval (synthetic-data-generator) "
        "cells, using trtr as their baseline",
    )
    args = parser.parse_args()

    df = load_long(args.csv)
    features, seed_counts = build_feature_matrix(
        df, metric=args.metric, include_tstr=args.include_tstr
    )

    features.to_csv(args.features_csv)
    print(f"\nSaved feature matrix to {args.features_csv}")
    print("\nFeature matrix (%-change-from-baseline):")
    print(features.round(1))

    dist = correlation_distance_matrix(
        features, strict_complete=args.strict_complete_cells
    )
    print("\nCorrelation distance matrix:")
    print(dist.round(3))

    Z = run_ward(dist)
    plot_dendrogram(
        Z,
        list(dist.index),
        args.out,
        title=f"Model architecture clustering ({args.metric}, {args.target})",
    )


if __name__ == "__main__":
    main()
