import sys
sys.path.insert(0, "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym")
from pathlib import Path

import pandas as pd
import torch
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset
from custom_graphgym.transform.normalization import denorm_ihs, denorm_log1p
from custom_graphgym.metric.regression import DisaggregationMetrics



# ── config ────────────────────────────────────────────────────────────────────
OUTPUT_DATA = Path(
    "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/output/earne_exp3"
)
raw_params = torch.load("datasets/earne/transform_single.pt")
DENORM_PARAMS = {
    "load_scaled": (raw_params["load"]["param1"], raw_params["load"]["param2"]),
    "pv_scaled": (raw_params["pv"]["param1"], raw_params["pv"]["param2"]),
    "net_scaled": (raw_params["net_demand"]["param1"], raw_params["net_demand"]["param2"]),
}


EXP_LABELS = {
    "earne_exp3-weather_mode=False-mp=0": r"W- MP0",
    "earne_exp3-weather_mode=False-mp=3": r"W- MP3",
    "earne_exp3-weather_mode=True-mp=0": r"W+ MP0",
    "earne_exp3-weather_mode=True-mp=3": r"W+ MP3",
}

BIN_EDGES = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0, float("inf")])
BIN_LABELS = {
    0: "No PV",
    1: "0%",
    2: "25%",
    3: "50%",
    4: "75%",
    5: "100%",
}
BIN_ORDER = list(BIN_LABELS.values()) + ["global"]


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


def compute_bin_indices(denorm_params):
    input_data = EARNeGraphDataset("./datasets/earne")
    node_pv_max = (
        denorm_ihs(input_data.pv_scaled, *denorm_params["pv_scaled"])
        * input_data.mask_data
    ).max(dim=0)[0]
    node_load_max = (
        denorm_log1p(input_data.load_scaled, *denorm_params["load_scaled"])
        * input_data.mask_data
    ).max(dim=0)[0]
    node_solar_penetration = node_pv_max / node_load_max
    return torch.bucketize(node_solar_penetration, BIN_EDGES)


# ── bin indices (shared across all experiments) ───────────────────────────────
bin_indices = compute_bin_indices(DENORM_PARAMS)

# ── pair pred/true files by experiment name ───────────────────────────────────
pred_files = {
    p.name.replace("_pred.pt", ""): p for p in OUTPUT_DATA.glob("*pred.pt")
}
true_files = {
    p.name.replace("_true.pt", ""): p for p in OUTPUT_DATA.glob("*true.pt")
}
experiments = sorted(pred_files.keys() & true_files.keys())
print(f"Found {len(experiments)} experiments: {experiments}")

# ── main loop ─────────────────────────────────────────────────────────────────

records = []

for exp_name in experiments:
    pred = denorm_pred(torch.load(pred_files[exp_name]), DENORM_PARAMS)
    true = denorm_true(torch.load(true_files[exp_name]), DENORM_PARAMS)

    bins_to_run = [("global", None)] + [
        (label, bin_id) for bin_id, label in BIN_LABELS.items()
    ]

    for label, bin_id in bins_to_run:
        if bin_id is None:
            metrics = DisaggregationMetrics.all(true, pred)
        else:
            mask = bin_indices == bin_id
            if mask.sum() == 0:
                continue
            metrics = DisaggregationMetrics.all(true[mask], pred[mask])

        for target, target_metrics in metrics.items():
            for metric, value in target_metrics.items():
                records.append(
                    {
                        "bin": label,
                        "metric": metric,
                        "target": target,
                        "experiment": EXP_LABELS.get(exp_name),
                        "value": value,
                    }
                )

# ── pivot into the right shape ────────────────────────────────────────────────
df_long = pd.DataFrame(records)

df = df_long.pivot_table(
    index=["bin", "metric"],
    columns=["target", "experiment"],
    values="value",
)

# enforce row order
METRIC_ORDER = [
    "rmse",
    "nrmse",
    "mae",
    "r2",
    "mbe",
    "efe",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "coverage",
    "sharpness",
    "net_rmse",
]
METRIC_LABELS = {
    "rmse": "RMSE",
    "nrmse": "NRMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "r2": "R$^2$",
    "mbe": "MBE",
    "efe": "EFE",
    "pinball_q10": "Pinball Q10",
    "pinball_q50": "Pinball Q50",
    "pinball_q90": "Pinball Q90",
    "coverage": "Coverage",
    "sharpness": "Sharpness",
    "net_rmse": "Net RMSE",
}


df = df.reindex(BIN_ORDER, level="bin").reindex(METRIC_ORDER, level="metric")
df.columns.names = ["target", "experiment"]
df.index.names = ["bin", "metric"]


LOWER_IS_BETTER = {
    "rmse",
    "nrmse",
    "mae",
    "mape",
    "mbe",
    "efe",
    "net rmse",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "sharpness",
}
HIGHER_IS_BETTER = {"r2", "coverage"}


def prune_metrics(df: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    """
    Remove specific metrics from the row index.

    Usage:
        df = prune_metrics(df, drop=["mape", "nrmse"])
    """
    keep = [m for m in METRIC_ORDER if m not in drop]
    available = df.index.get_level_values("metric").unique()
    keep = [m for m in keep if m in available]
    return df.loc[pd.IndexSlice[:, keep], :]


drop_metrics = [
    "nrmse",
    "mbe",
    "pinball_q10",
    "pinball_q90",
    "pinball_q50" "net_rmse",
]

df = prune_metrics(df, drop_metrics)

df = df.rename(index=METRIC_LABELS, level="metric")
print(df)


def prepare_df(
    df: pd.DataFrame, drop_metrics: list[str] | None = None
) -> pd.DataFrame:
    """Rename experiments + metrics, prune, reindex, round."""
    df = df.rename(columns=EXP_LABELS, level="experiment")
    df = df.rename(index=METRIC_LABELS, level="metric")

    if drop_metrics:
        # translate to labels if raw names passed
        drop_labels = [METRIC_LABELS.get(m, m) for m in drop_metrics]
        available = df.index.get_level_values("metric").unique()
        df = df.drop(
            index=[m for m in drop_labels if m in available], level="metric"
        )

    # enforce metric row order
    ordered_labels = [
        METRIC_LABELS[m]
        for m in METRIC_ORDER
        if METRIC_LABELS.get(m, m) in df.index.get_level_values("metric")
    ]
    df = df.reindex(ordered_labels, level="metric")

    return df.round(2)


def highlight_best(df: pd.DataFrame):
    def bold_best(series: pd.Series, lower_is_better: bool) -> list[str]:
        if series.isna().all():
            return [""] * len(series)
        best = series.min() if lower_is_better else series.max()
        return ["font-weight: bold" if v == best else "" for v in series]

    styler = df.style

    for target in df.columns.get_level_values("target").unique():
        target_cols = df.columns[
            df.columns.get_level_values("target") == target
        ]

        for bin_label in df.index.get_level_values("bin").unique():
            for metric in df.index.get_level_values("metric").unique():
                if (bin_label, metric) not in df.index:
                    continue
                if df.loc[(bin_label, metric), target_cols].isna().all():
                    continue

                # map display label back to raw metric key for direction lookup
                raw_metric = next(
                    (k for k, v in METRIC_LABELS.items() if v == metric), None
                )
                lower = raw_metric in LOWER_IS_BETTER if raw_metric else True

                styler = styler.apply(
                    bold_best,
                    lower_is_better=lower,
                    subset=pd.IndexSlice[(bin_label, metric), target_cols],
                    axis=1,
                )

    return styler


def export_latex(df, path):
    df = df.rename(index=lambda x: x.replace("%", r"\%"), level="bin")

    highlight_best(df).format("{:.2f}", na_rep="---").to_latex(
        buf=path,
        hrules=True,
        convert_css=True,
        caption="Disaggregation metrics by solar penetration bin",
        label="tab:disagg_metrics",
        position="htbp",
    )

    # add \midrule between bins
    with open(path, "r") as f:
        latex = f.read()

    latex = latex.replace(r"\multirow", r"\midrule" + "\n" + r"\multirow")

    # remove the first \midrule we just added (it's right after \midrule from hrules)
    latex = latex.replace(r"\midrule" + "\n" + r"\midrule", r"\midrule", 1)

    with open(path, "w") as f:
        f.write(latex)

    print(f"saved → {path}")


export_latex(df, "results.tex")
