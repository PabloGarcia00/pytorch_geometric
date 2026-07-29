import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import custom_graphgym  # noqa — registers loaders/losses/metrics
import pandas as pd
import torch
import tqdm
import yaml
from custom_graphgym.metric.regression import DisaggregationMetrics
from custom_graphgym.target_utils import active_targets

from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path("results")


# ── discovery ─────────────────────────────────────────────────────────────────
def _load_last_stats(path: Path) -> dict:
    """stats.json files are JSON-lines (one record per epoch); return the last one."""
    if not path.exists():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def _parse_run_tag(run_name: str) -> dict:
    """'earne_learned_corr_dual-span=False-wx=True' -> {'model_name': 'earne_learned_corr_dual', 'span': False, 'wx': True}"""
    base, *params = run_name.split("-")
    tag: dict[str, bool | str] = {"model_name": base}
    for param in params:
        if "=" not in param:
            continue
        key, value = param.split("=", 1)
        tag[key] = value if value not in ("True", "False") else value == "True"
    return tag


def discover_manifest(results_root: Path | str = RESULTS_ROOT) -> pd.DataFrame:
    """
    Walk `results_root/<sweep>/<run>/<seed>/` and return one row per (sweep, run, seed)
    pointing at that replicate's config, checkpoint, and stats — the starting point for
    running/aggregating metrics generically across every disaggregation model in `results/`.
    """
    results_root = Path(results_root)
    records = []

    for config_path in sorted(results_root.glob("*/*/config.yaml")):
        run_dir = config_path.parent
        cfg_yaml = yaml.safe_load(config_path.read_text())
        dataset_cfg = cfg_yaml.get("dataset", {})
        earne_cfg = cfg_yaml.get("earne_data", {})
        model_cfg = cfg_yaml.get("model", {})

        seed_dirs = sorted(
            (p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )

        for seed_dir in seed_dirs:
            ckpts = sorted(seed_dir.glob("ckpt/*.ckpt"))
            test_stats = _load_last_stats(seed_dir / "test" / "stats.json")
            val_stats = _load_last_stats(seed_dir / "val" / "stats.json")

            records.append(
                {
                    "sweep": run_dir.parent.name,
                    "run_name": run_dir.name,
                    **_parse_run_tag(run_dir.name),
                    "seed": int(seed_dir.name),
                    "run_dir": run_dir,
                    "seed_dir": seed_dir,
                    "config_path": config_path,
                    "ckpt_path": ckpts[-1] if ckpts else None,
                    "n_ckpts": len(ckpts),
                    "dataset_dir": dataset_cfg.get("dir"),
                    "processed_root": earne_cfg.get("processed_root"),
                    # the loader reads/writes datasets under earne_data.processed_root,
                    # not dataset.dir (which is unused by earne_loader_new) — check that
                    "dataset_exists": (
                        Path(earne_cfg.get("processed_root", "")) / "processed" / "data.pt"
                    ).exists(),
                    "dataset_format": dataset_cfg.get("format"),
                    "graph_mode": earne_cfg.get("graph_mode"),
                    "dual_read": earne_cfg.get("dual_read"),
                    "weather_mode": earne_cfg.get("weather_mode"),
                    "mask_physics_impossible": earne_cfg.get("mask_physics_impossible"),
                    "require_full_span": earne_cfg.get("require_full_span"),
                    "model_type": model_cfg.get("type"),
                    "dim_in": model_cfg.get("dim_in"),
                    "n_quantiles": model_cfg.get("n_quantiles"),
                    # absent in configs predating cfg.model.predict_targets
                    # -- those were always dual-target (load + PV)
                    "predict_targets": model_cfg.get("predict_targets", ["load", "pv"]),
                    "test_loss": test_stats.get("loss"),
                    "test_mae_load": test_stats.get("earne_mae_load"),
                    "test_mae_pv": test_stats.get("earne_mae_pv"),
                    "val_loss": val_stats.get("loss"),
                }
            )

    return pd.DataFrame(records)


# ── disaggregation (inference on the test split) ───────────────────────────────
# Model types with no trainable parameters -- "fitting" is a deterministic,
# no-grad cache population over the training split (see
# custom_graphgym/network/baseline_knn.py's fit_cache), not something saved
# to a checkpoint -- so there's no ckpt_path to reload here. Rebuilding the
# cache from the training split is cheap and gives identical results to
# whatever the original training run cached.
CACHE_FIT_MODEL_TYPES = {"baseline_knn"}


def _load_state_dict(ckpt_path: Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if model_keys[0].startswith("model.") and not ckpt_keys[0].startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif ckpt_keys[0].startswith("model.") and not model_keys[0].startswith("model."):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def disaggregate_test_set(
    row: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list] | None:
    """
    Load `row`'s config + checkpoint, run inference over its test split, and
    denormalize predictions/labels back into physical units (W).

    Returns (true, pred, user_ids, timestamps):
        true: [N, 4, T]  — y_load, y_pv, mask, y_net_demand (always both,
            regardless of what was predicted)
        pred: [N, n_quantiles * len(targets), T] — one quantile block per
            cfg.model.predict_targets entry, in active_targets() order
        user_ids: length-N list, aligned with true/pred's node dimension
        timestamps: length-T list of the real timestamp each T column
            corresponds to. The test loader uses shuffle=False (see
            torch_geometric.graphgym.loader.create_loader), so batches are
            iterated in ascending test-split order — recomputing
            get_split_indices() here reproduces that same order without
            depending on dataset.data.test_graph_index, which create_loader
            deletes once the loader is built.
    or None if the dataset/checkpoint for this run isn't available on this machine.
    """
    cache_fit = row["model_type"] in CACHE_FIT_MODEL_TYPES
    if row["ckpt_path"] is None and not cache_fit:
        return None

    cfg.set_new_allowed(True)
    load_cfg(cfg, argparse.Namespace(cfg_file=str(row["config_path"]), opts=[]))
    # this script does one no-grad pass over an already-processed dataset;
    # background loader workers add nothing here and would multiply with this
    # script's own process-pool parallelism, oversubscribing the host
    cfg.num_workers = 0
    cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.accelerator)
    targets = active_targets()
    n_q = cfg.model.n_quantiles

    datamodule = GraphGymDataModule()
    test_loader = datamodule.test_dataloader()
    dataset = test_loader.dataset

    model = create_model()
    if cache_fit:
        model.model.fit_cache(datamodule.train_dataloader())
    else:
        _load_state_dict(row["ckpt_path"], model)
    model = model.to(device).eval()

    true_chunks, pred_chunks = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred, true = model(batch)  # (pred, true) per GraphGymModule.forward()
            # pred: [N, n_quantiles * len(targets)]
            # true: [N, 4]                (y_load, y_pv, mask, y_net_demand)

            t_per_batch = pred.shape[0] // dataset.num_nodes
            pred_chunks.append(
                pred.reshape(dataset.num_nodes, t_per_batch, n_q * len(targets))
                .permute(0, 2, 1)  # → [num_nodes, n_q * len(targets), t_per_batch]
                .cpu()
            )
            true_chunks.append(
                true.reshape(dataset.num_nodes, t_per_batch, 4)
                .permute(0, 2, 1)  # → [num_nodes, 4, t_per_batch]
                .cpu()
            )

    pred = torch.cat(pred_chunks, dim=2)
    true = torch.cat(true_chunks, dim=2)

    t = dataset.transform_obj
    pred_denorm = torch.cat(
        [
            t.inverse_transform(name, pred[:, i * n_q : (i + 1) * n_q])
            for i, name in enumerate(targets)
        ],
        dim=1,
    )
    true_denorm = torch.cat(
        [
            t.inverse_transform("load", true[:, [0]]),
            t.inverse_transform("pv", true[:, [1]]),
            true[:, [2]],  # mask — not a normalized stream
            t.inverse_transform("net_demand", true[:, [3]]),
        ],
        dim=1,
    )
    _, _, test_idx = dataset.get_split_indices()
    timestamps = [dataset.timestamps[i + dataset.seq_len] for i in test_idx]

    return true_denorm, pred_denorm, dataset.active_ids, timestamps


def compute_metrics(row: pd.Series) -> tuple[dict, list[dict]] | None:
    """
    Disaggregate `row`'s test set once and score it two ways from the same
    (true, pred) tensors: pooled across all households (compact summary,
    -> metrics_long.csv) and per household (drill-down, -> metrics_per_household.csv).
    Scoring both from a single disaggregate_test_set() call avoids re-running
    the expensive part (config/dataset/model load + inference) a second time
    just to get a per-household breakdown.
    """
    result = disaggregate_test_set(row)
    if result is None:
        return None
    true, pred, user_ids, _ = result
    # cfg still reflects `row`'s config -- disaggregate_test_set's load_cfg
    # call mutated the module-level singleton and nothing has touched it
    # since (this call happens immediately after, synchronously).
    targets = active_targets()
    n_q = cfg.model.n_quantiles
    q50_idx = cfg.model.quantiles.index(0.5)

    pooled = DisaggregationMetrics.all(
        true.numpy(), pred.numpy(), targets=targets, n_quantiles=n_q, q50_idx=q50_idx,
    )

    household_records = []
    for i, uid in enumerate(user_ids):
        household_metrics = DisaggregationMetrics.all(
            true[i : i + 1].numpy(), pred[i : i + 1].numpy(),
            targets=targets, n_quantiles=n_q, q50_idx=q50_idx,
        )
        for target, target_metrics in household_metrics.items():
            for metric, value in target_metrics.items():
                household_records.append(
                    {
                        "sweep": row["sweep"],
                        "run_name": row["run_name"],
                        "seed": row["seed"],
                        "user_id": uid,
                        "target": target,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pooled, household_records


# ── parallel execution ──────────────────────────────────────────────────────
def _compute_metrics_worker(
    row: pd.Series,
) -> tuple[pd.Series, dict | None, list[dict] | None, str | None]:
    """Picklable ProcessPoolExecutor wrapper: returns (row, pooled, household_records,
    error) instead of raising, so the parent can preserve the existing
    FAILED-and-continue behavior."""
    try:
        result = compute_metrics(row)
    except Exception as e:
        return row, None, None, str(e)
    if result is None:
        return row, None, None, None
    pooled, household_records = result
    return row, pooled, household_records, None


def _init_worker() -> None:
    """Runs once per spawned worker before any task: forces CPU-only inference
    (avoids GPU contention/OOM across workers sharing the same GPU(s)) and caps
    intra-op/BLAS threading so `--workers` processes don't each try to claim
    every core on the host."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _metrics_to_records(row: pd.Series, metrics: dict, order: int) -> list[dict]:
    """Expand one row's compute_metrics() output into metrics_long.csv records —
    shared by both the serial and parallel __main__ branches."""
    return [
        {
            "sweep": row["sweep"],
            "run_name": row["run_name"],
            "seed": row["seed"],
            "target": target,
            "metric": metric,
            "value": value,
            "_order": order,
        }
        for target, target_metrics in metrics.items()
        for metric, value in target_metrics.items()
    ]


# ── metric selection ─────────────────────────────────────────────────────────
METRIC_ORDER = [
    "rmse",
    "nrmse",
    "rmse_maxnorm",
    "mae",
    "nmae",
    "r2",
    "mbe",
    "nmbe",
    "efe",
    "pinball_q10",
    "npinball_q10",
    "pinball_q50",
    "npinball_q50",
    "pinball_q90",
    "npinball_q90",
    "coverage",
    "sharpness",
    "nsharpness",
    "net_rmse",
    "nnet_rmse",
    "export_violation_rate",
    "export_violation_count",
    "export_violation_mag",
    "nexport_violation_mag",
]
METRIC_LABELS = {
    "rmse": "RMSE",
    "nrmse": "NRMSE",
    "rmse_maxnorm": "RMSE / max",
    "mae": "MAE",
    "nmae": "NMAE",
    "mape": "MAPE",
    "r2": "R$^2$",
    "mbe": "MBE",
    "nmbe": "NMBE",
    "efe": "EFE",
    "pinball_q10": "Pinball Q10",
    "npinball_q10": "NPinball Q10",
    "pinball_q50": "Pinball Q50",
    "npinball_q50": "NPinball Q50",
    "pinball_q90": "Pinball Q90",
    "npinball_q90": "NPinball Q90",
    "coverage": "Coverage",
    "sharpness": "Sharpness",
    "nsharpness": "NSharpness",
    "net_rmse": "Net RMSE",
    "nnet_rmse": "Net NRMSE",
    "export_violation_rate": "PV<Export Rate",
    "export_violation_count": "PV<Export Count",
    "export_violation_mag": "PV<Export Mag (W)",
    "nexport_violation_mag": "PV<Export NMag",
}
LOWER_IS_BETTER = {
    "rmse", "nrmse", "rmse_maxnorm", "mae", "nmae", "mape",
    "mbe", "nmbe", "efe",
    "net_rmse", "nnet_rmse",
    "pinball_q10", "npinball_q10",
    "pinball_q50", "npinball_q50",
    "pinball_q90", "npinball_q90",
    "sharpness", "nsharpness",
    "export_violation_rate", "export_violation_count",
    "export_violation_mag", "nexport_violation_mag",
}
HIGHER_IS_BETTER = {"r2", "coverage"}

# metrics to hide from the final table — still computed, just noisy/redundant to show
DROP_METRICS = [
    "rmse_maxnorm", "mbe", "nmbe",
    "pinball_q10", "npinball_q10",
    "pinball_q90", "npinball_q90",
]


def prune_metrics(df: pd.DataFrame, drop: list[str] = DROP_METRICS) -> pd.DataFrame:
    """Keep only METRIC_ORDER minus `drop`, in canonical order."""
    keep = [m for m in METRIC_ORDER if m not in drop and m in df.index]
    return df.loc[keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Process-pool size for parallel disaggregation (1 = legacy serial "
             "path, still auto-detects CUDA; >1 forces CPU-only workers to avoid "
             "GPU contention).",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    manifest = discover_manifest()
    print(f"Discovered {len(manifest)} runs across {manifest['sweep'].nunique()} sweeps")

    records = []
    household_records = []
    runnable = manifest[
        manifest["ckpt_path"].notna()
        | manifest["model_type"].isin(CACHE_FIT_MODEL_TYPES)
    ]
    skipped = len(manifest) - len(runnable)
    if skipped:
        print(f"Skipping {skipped} runs — no checkpoint")

    if args.workers <= 1:
        for order, (_, row) in enumerate(
            tqdm.tqdm(runnable.iterrows(), total=len(runnable), desc="disaggregating test sets")
        ):
            try:
                result = compute_metrics(row)
            except Exception as e:
                print(f"FAILED  {row['sweep']}/{row['run_name']}/seed={row['seed']}: {e}")
                continue
            if result is None:
                continue
            pooled, hh_records = result
            records.extend(_metrics_to_records(row, pooled, order))
            household_records.extend(hh_records)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx, initializer=_init_worker
        ) as executor:
            future_to_order = {
                executor.submit(_compute_metrics_worker, row): order
                for order, (_, row) in enumerate(runnable.iterrows())
            }
            for future in tqdm.tqdm(
                as_completed(future_to_order), total=len(future_to_order),
                desc="disaggregating test sets",
            ):
                row, pooled, hh_records, error = future.result()
                if error is not None:
                    print(f"FAILED  {row['sweep']}/{row['run_name']}/seed={row['seed']}: {error}")
                    continue
                if pooled is None:
                    continue
                records.extend(_metrics_to_records(row, pooled, future_to_order[future]))
                household_records.extend(hh_records)

    df_long = pd.DataFrame(records)
    if df_long.empty:
        print("No runs were disaggregated — nothing to report.")
    else:
        df_long = df_long.sort_values(["_order", "target", "metric"]).drop(columns="_order")
        df_long.to_csv("notebooks/metrics_long.csv", index=False)

        # wide: mean-aggregated across seeds (one column per target/sweep/run_name)
        df = df_long.pivot_table(
            index="metric",
            columns=["target", "sweep", "run_name"],
            values="value",
            aggfunc="mean",
        )
        df = prune_metrics(df)
        df = df.rename(index=METRIC_LABELS)
        df.to_csv("notebooks/metrics_wide.csv")
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        print(df)

    df_household = pd.DataFrame(household_records)
    if not df_household.empty:
        df_household = df_household.sort_values(
            ["sweep", "run_name", "seed", "user_id", "target", "metric"]
        )
        df_household.to_csv("notebooks/metrics_per_household.csv", index=False)
        print(
            f"[OK] wrote notebooks/metrics_per_household.csv ({len(df_household)} rows, "
            f"{df_household['run_name'].nunique()} runs, {df_household['user_id'].nunique()} households)"
        )
