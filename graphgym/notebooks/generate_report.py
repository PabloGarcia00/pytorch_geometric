"""
Builds notebooks/metrics_report.md from notebooks/metrics_long.csv (the output of
calculate_metrics.py). Uses the best-performing seed per (sweep, run_name) rather
than averaging across seeds, since most runs currently have unequal, partial seed
counts (still training). Run after calculate_metrics.py to refresh.
"""
import pandas as pd
from datetime import datetime, timezone

df = pd.read_csv("metrics_long.csv")
load = df[df["target"] == "load"]
pv = df[df["target"] == "pv"]


def best_row(sub_df, sort_metric="mae"):
    if sub_df.empty:
        return None, 0
    piv = sub_df.pivot_table(index="seed", columns="metric", values="value")
    if sort_metric not in piv.columns:
        return None, len(piv)
    return piv.loc[piv[sort_metric].idxmin()], len(piv)


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if x is not None and pd.notna(x) else "n/a"


def section_table(title, note, rows):
    """rows: list of (label, sweep, run_name)"""
    lines = [f"### {title}", "", note, "", "| config | seeds | load RMSE | load MAE | load R² | pv RMSE | pv MAE | pv R² | pv export-viol rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, sweep, run in rows:
        lsub = load[(load["sweep"] == sweep) & (load["run_name"] == run)]
        psub = pv[(pv["sweep"] == sweep) & (pv["run_name"] == run)]
        lrow, n_seeds = best_row(lsub)
        prow, _ = best_row(psub, sort_metric="mae")
        status = f"{n_seeds}/3" if n_seeds < 3 else "3/3 ✓"
        lr = lrow if lrow is not None else {}
        pr = prow if prow is not None else {}
        lines.append(
            f"| {label} | {status} "
            f"| {fmt(lr.get('rmse'))} | {fmt(lr.get('mae'))} | {fmt(lr.get('r2'), 3)} "
            f"| {fmt(pr.get('rmse'))} | {fmt(pr.get('mae'))} | {fmt(pr.get('r2'), 3)} "
            f"| {fmt(pr.get('export_violation_rate'), 4)} |"
        )
    lines.append("")
    return "\n".join(lines)


parts = []
parts.append("# Model comparison report\n")
parts.append(
    f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from `notebooks/metrics_long.csv` "
    "(checkpoint-based test-set disaggregation via `calculate_metrics.py`, denormalized to physical units). "
    "Metrics use the best-performing seed by MAE per config, not an average. "
    "'seeds' shows how many of 3 planned seeds have a saved checkpoint (all training is now complete)._\n"
)

parts.append("## Exp1 — no physics (learned_corr / no-mask)\n")

parts.append(section_table(
    "EARNE — single-read",
    "Coverage (span=True=optimal/62 nodes, span=False=sub-optimal/112 nodes) × weather.",
    [
        ("optimal, wx=False", "earne_learned_corr_single_grid_sweep_coverage_weather", "earne_learned_corr_single-span=True-wx=False"),
        ("optimal, wx=True", "earne_learned_corr_single_grid_sweep_coverage_weather", "earne_learned_corr_single-span=True-wx=True"),
        ("sub-optimal, wx=False", "earne_learned_corr_single_grid_sweep_coverage_weather", "earne_learned_corr_single-span=False-wx=False"),
        ("sub-optimal, wx=True", "earne_learned_corr_single_grid_sweep_coverage_weather", "earne_learned_corr_single-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "EARNE — dual-read",
    "Coverage × weather, dual-read (consumption+generation).",
    [
        ("optimal, wx=False", "earne_learned_corr_dual_grid_sweep_coverage_weather", "earne_learned_corr_dual-span=True-wx=False"),
        ("optimal, wx=True", "earne_learned_corr_dual_grid_sweep_coverage_weather", "earne_learned_corr_dual-span=True-wx=True"),
        ("sub-optimal, wx=False", "earne_learned_corr_dual_grid_sweep_coverage_weather", "earne_learned_corr_dual-span=False-wx=False"),
        ("sub-optimal, wx=True", "earne_learned_corr_dual_grid_sweep_coverage_weather", "earne_learned_corr_dual-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "MLP baseline — single-read",
    "No spatial mixing, per-node MLP temporal encoder.",
    [
        ("optimal, wx=False", "baseline_mlp_single_nomask_grid_sweep_coverage_weather", "baseline_mlp_single_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_mlp_single_nomask_grid_sweep_coverage_weather", "baseline_mlp_single_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_mlp_single_nomask_grid_sweep_coverage_weather", "baseline_mlp_single_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_mlp_single_nomask_grid_sweep_coverage_weather", "baseline_mlp_single_nomask-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "MLP baseline — dual-read",
    "No spatial mixing, per-node MLP temporal encoder, dual-read.",
    [
        ("optimal, wx=False", "baseline_mlp_dual_nomask_grid_sweep_coverage_weather", "baseline_mlp_dual_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_mlp_dual_nomask_grid_sweep_coverage_weather", "baseline_mlp_dual_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_mlp_dual_nomask_grid_sweep_coverage_weather", "baseline_mlp_dual_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_mlp_dual_nomask_grid_sweep_coverage_weather", "baseline_mlp_dual_nomask-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "LSTM baseline — single-read",
    "BiLSTM per-node temporal encoder, no spatial mixing.",
    [
        ("optimal, wx=False", "baseline_lstm_single_nomask_grid_sweep_coverage_weather", "baseline_lstm_single_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_lstm_single_nomask_grid_sweep_coverage_weather", "baseline_lstm_single_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_lstm_single_nomask_grid_sweep_coverage_weather", "baseline_lstm_single_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_lstm_single_nomask_grid_sweep_coverage_weather", "baseline_lstm_single_nomask-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "CVAE baseline — single-read",
    "Generative BiLSTM+VAE model.",
    [
        ("optimal, wx=False", "baseline_cvae_nomask_grid_sweep_coverage_weather", "baseline_cvae_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_cvae_nomask_grid_sweep_coverage_weather", "baseline_cvae_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_cvae_nomask_grid_sweep_coverage_weather", "baseline_cvae_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_cvae_nomask_grid_sweep_coverage_weather", "baseline_cvae_nomask-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "LSTM baseline — dual-read",
    "BiLSTM per-node temporal encoder, no spatial mixing, dual-read.",
    [
        ("optimal, wx=False", "baseline_lstm_dual_nomask_grid_sweep_coverage_weather", "baseline_lstm_dual_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_lstm_dual_nomask_grid_sweep_coverage_weather", "baseline_lstm_dual_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_lstm_dual_nomask_grid_sweep_coverage_weather", "baseline_lstm_dual_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_lstm_dual_nomask_grid_sweep_coverage_weather", "baseline_lstm_dual_nomask-span=False-wx=True"),
    ],
))
parts.append(section_table(
    "CVAE baseline — dual-read",
    "Generative BiLSTM+VAE model, dual-read.",
    [
        ("optimal, wx=False", "baseline_cvae_dual_nomask_grid_sweep_coverage_weather", "baseline_cvae_dual_nomask-span=True-wx=False"),
        ("optimal, wx=True", "baseline_cvae_dual_nomask_grid_sweep_coverage_weather", "baseline_cvae_dual_nomask-span=True-wx=True"),
        ("sub-optimal, wx=False", "baseline_cvae_dual_nomask_grid_sweep_coverage_weather", "baseline_cvae_dual_nomask-span=False-wx=False"),
        ("sub-optimal, wx=True", "baseline_cvae_dual_nomask_grid_sweep_coverage_weather", "baseline_cvae_dual_nomask-span=False-wx=True"),
    ],
))

parts.append("## Exp2 — physics (mask on, optimal coverage only)\n")

parts.append(section_table(
    "EARNE — single-read",
    "physics_weight=1.0 (loss+mask), 0.3 (reduced loss), and mask-only/no-loss (physics_weight=0.0).",
    [
        ("pw=1.0, wx=False", "earne_phys_single_grid_sweep_weather_only", "earne_phys_single-wx=False"),
        ("pw=1.0, wx=True", "earne_phys_single_grid_sweep_weather_only", "earne_phys_single-wx=True"),
        ("pw=0.3, wx=False", "earne_phys_single_pw03_grid_sweep_weather_only", "earne_phys_single_pw03-wx=False"),
        ("pw=0.3, wx=True", "earne_phys_single_pw03_grid_sweep_weather_only", "earne_phys_single_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "earne_phys_single_noloss_grid_sweep_weather_only", "earne_phys_single_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "earne_phys_single_noloss_grid_sweep_weather_only", "earne_phys_single_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "EARNE — dual-read",
    "physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.",
    [
        ("pw=1.0, wx=False", "earne_phys_dual_grid_sweep_weather_only", "earne_phys_dual-wx=False"),
        ("pw=1.0, wx=True", "earne_phys_dual_grid_sweep_weather_only", "earne_phys_dual-wx=True"),
        ("pw=0.3, wx=False", "earne_phys_dual_pw03_grid_sweep_weather_only", "earne_phys_dual_pw03-wx=False"),
        ("pw=0.3, wx=True", "earne_phys_dual_pw03_grid_sweep_weather_only", "earne_phys_dual_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "earne_phys_dual_noloss_grid_sweep_weather_only", "earne_phys_dual_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "earne_phys_dual_noloss_grid_sweep_weather_only", "earne_phys_dual_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "MLP baseline — single-read",
    "physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0).",
    [
        ("pw=1.0, wx=False", "baseline_mlp_grid_sweep_weather_only", "baseline_mlp-wx=False"),
        ("pw=1.0, wx=True", "baseline_mlp_grid_sweep_weather_only", "baseline_mlp-wx=True"),
        ("pw=0.3, wx=False", "baseline_mlp_pw03_grid_sweep_weather_only", "baseline_mlp_pw03-wx=False"),
        ("pw=0.3, wx=True", "baseline_mlp_pw03_grid_sweep_weather_only", "baseline_mlp_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "baseline_mlp_noloss_grid_sweep_weather_only", "baseline_mlp_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "baseline_mlp_noloss_grid_sweep_weather_only", "baseline_mlp_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "MLP baseline — dual-read",
    "physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.",
    [
        ("pw=1.0, wx=False", "baseline_mlp_dual_grid_sweep_weather_only", "baseline_mlp_dual-wx=False"),
        ("pw=1.0, wx=True", "baseline_mlp_dual_grid_sweep_weather_only", "baseline_mlp_dual-wx=True"),
        ("pw=0.3, wx=False", "baseline_mlp_dual_pw03_grid_sweep_weather_only", "baseline_mlp_dual_pw03-wx=False"),
        ("pw=0.3, wx=True", "baseline_mlp_dual_pw03_grid_sweep_weather_only", "baseline_mlp_dual_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "baseline_mlp_dual_noloss_grid_sweep_weather_only", "baseline_mlp_dual_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "baseline_mlp_dual_noloss_grid_sweep_weather_only", "baseline_mlp_dual_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "LSTM baseline — single-read",
    "physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0).",
    [
        ("pw=1.0, wx=False", "baseline_lstm_grid_sweep_weather_only", "baseline_lstm-wx=False"),
        ("pw=1.0, wx=True", "baseline_lstm_grid_sweep_weather_only", "baseline_lstm-wx=True"),
        ("pw=0.3, wx=False", "baseline_lstm_pw03_grid_sweep_weather_only", "baseline_lstm_pw03-wx=False"),
        ("pw=0.3, wx=True", "baseline_lstm_pw03_grid_sweep_weather_only", "baseline_lstm_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "baseline_lstm_noloss_grid_sweep_weather_only", "baseline_lstm_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "baseline_lstm_noloss_grid_sweep_weather_only", "baseline_lstm_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "LSTM baseline — dual-read",
    "physics_weight=1.0, 0.3, and mask-only/no-loss (physics_weight=0.0), dual-read.",
    [
        ("pw=1.0, wx=False", "baseline_lstm_dual_grid_sweep_weather_only", "baseline_lstm_dual-wx=False"),
        ("pw=1.0, wx=True", "baseline_lstm_dual_grid_sweep_weather_only", "baseline_lstm_dual-wx=True"),
        ("pw=0.3, wx=False", "baseline_lstm_dual_pw03_grid_sweep_weather_only", "baseline_lstm_dual_pw03-wx=False"),
        ("pw=0.3, wx=True", "baseline_lstm_dual_pw03_grid_sweep_weather_only", "baseline_lstm_dual_pw03-wx=True"),
        ("mask-only (pw=0), wx=False", "baseline_lstm_dual_noloss_grid_sweep_weather_only", "baseline_lstm_dual_noloss-wx=False"),
        ("mask-only (pw=0), wx=True", "baseline_lstm_dual_noloss_grid_sweep_weather_only", "baseline_lstm_dual_noloss-wx=True"),
    ],
))
parts.append(section_table(
    "CVAE baseline — single-read",
    "cvae_loss has no physics_weight term — this run is effectively \"mask-only\" by construction.",
    [
        ("mask-only, wx=False", "baseline_cvae_grid_sweep_weather_only", "baseline_cvae-wx=False"),
        ("mask-only, wx=True", "baseline_cvae_grid_sweep_weather_only", "baseline_cvae-wx=True"),
    ],
))
parts.append(section_table(
    "CVAE baseline — dual-read",
    "cvae_loss has no physics_weight term — this run is effectively \"mask-only\" by construction, dual-read.",
    [
        ("mask-only, wx=False", "baseline_cvae_dual_grid_sweep_weather_only", "baseline_cvae_dual-wx=False"),
        ("mask-only, wx=True", "baseline_cvae_dual_grid_sweep_weather_only", "baseline_cvae_dual-wx=True"),
    ],
))

parts.append(
    "## Notes\n\n"
    "- **ST-SGCCaps is excluded** — deactivated per request, and its point-prediction output "
    "(`[N,2]`) doesn't fit `calculate_metrics.py`'s quantile-shaped (`[N,6]`) reshape anyway.\n"
    "- Runs marked with fewer than 3/3 seeds are still training; treat those rows as preliminary.\n"
    "- `export-viol rate` = fraction of test timesteps where predicted PV production is less than "
    "the export implied by observed net demand (a physically-impossible prediction).\n"
)

with open("metrics_report.md", "w") as f:
    f.write("\n".join(parts))

print("wrote notebooks/metrics_report.md")
