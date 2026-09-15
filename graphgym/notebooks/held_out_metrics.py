"""
Scores trained checkpoints against households EXCLUDED by
cfg.earne_data.require_full_span -- a genuine held-out population never
seen in that run's train, val, or test split -- to measure how well each
model generalizes to households it never trained on.

require_full_span (default True) drops any household whose observations
don't span the first and last week of the dataset's time range
(custom_graphgym/loader/graph_dataset.py's _get_node_mask()). That check
runs after the cached data.pt bundle is loaded, never during process(), so
reloading the same processed_root with require_full_span=False is cheap
(no reprocessing) and yields the full native household universe to diff
against -- see resolve_held_out_ids().

Held-out households are scored over the same val+test window used by
calculate_metrics.py's disaggregate_test_set() -- NOT the full available
timeline. This dataset's chronological 70/15/15 split puts "test" in
late-Oct-through-early-Apr (no full summer); "val" is never used for
gradient updates (only best-val-loss checkpoint selection) and does cover
a full summer, so folding it in gives a season-balanced, leak-safe window.
Held-out households were never used in ANY split, so any window is
leakage-free for them -- using the SAME val+test window as the seen
evaluation (rather than their full timeline) is what makes the two
comparable: the seen and held-out numbers must average over the same
calendar period, or a season/signal-level mismatch (e.g. winter-only vs.
winter+summer) can dominate the apparent generalization gap in either
direction.

Usage:
    python notebooks/held_out_metrics.py
    python notebooks/held_out_metrics.py --workers 64
"""
import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Must be set before polars' first import -- see calculate_metrics.py's
# matching line for why (polars defaults its async I/O thread pool to
# nproc per process with no cap, which oversubscribes the host and panics
# under N concurrent workers). Technically inherited transitively via the
# `from calculate_metrics import ...` below (that module sets this before
# its own `import custom_graphgym`), but set explicitly here too so this
# file doesn't depend on that import ordering to stay correct.
os.environ.setdefault("POLARS_MAX_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import tqdm
from calculate_metrics import (
    METRIC_LABELS,
    ROW_TIMEOUT_S,
    RowTimeout,
    _init_worker,
    _load_state_dict,
    _strat_fields,
    discover_manifest,
    prune_metrics,
    with_row_timeout,
)

import custom_graphgym  # noqa — registers loaders/losses/metrics
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset
from custom_graphgym.metric.regression import DisaggregationMetrics
from custom_graphgym.target_utils import active_targets

from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.loader import create_dataset
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.loader import DataLoader as PyGDataLoader

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

# Architectures that bake the trained node population into learned
# parameters -- structurally incapable of scoring a household absent from
# training. Deny-list (not allow-list) so new node-agnostic architectures
# don't need this script updated.
NODE_LOCKED_MODEL_TYPES = {
    "baseline_linear",  # _baseline_common.py PerNodeLinearQuantileHead:
                        # per-node weight/bias [num_nodes, in, out]
    "baseline_svr",     # same head
    "baseline_knn",     # dense per-node neighbor cache [num_nodes, S_train, F]
    "st_sgc_caps",      # CapsuleRegressor(n_sites=cfg.st_caps.n_user), fixed
                        # at construction
}


# ── held-out population ─────────────────────────────────────────────────────
def resolve_held_out_ids(row: pd.Series) -> list[str]:
    """
    IDs present in the native (require_full_span=False) dataset but
    dropped by this run's own require_full_span filtering -- households
    the model never saw in train, val, or test.

    Only require_full_span is flipped between the two loads --
    filter_zips/filter_ids from the row's own config are left untouched,
    so the diff isolates exactly the temporal-span filter's effect within
    whatever population scope that run already intended.
    """
    # load_cfg() only does cfg.merge_from_file() -- it never resets to
    # defaults -- so a previous row's mutations (below, and in
    # disaggregate_held_out_set) would otherwise leak into this row.
    cfg.set_new_allowed(True)
    cfg.earne_data.filter_ids = []
    cfg.earne_data.require_full_span = True
    load_cfg(cfg, argparse.Namespace(cfg_file=str(row["config_path"]), opts=[]))

    trained_ids = set(
        EARNeGraphDataset(
            root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len
        ).active_ids
    )
    cfg.earne_data.require_full_span = False
    full_ids = set(
        EARNeGraphDataset(
            root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len
        ).active_ids
    )
    return sorted(full_ids - trained_ids)


# ── disaggregation (inference over the held-out set's full timeline) ───────
def disaggregate_held_out_set(
    row: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int] | None:
    """
    Load `row`'s config + checkpoint, run inference over the SAME val+test
    window as calculate_metrics.py's disaggregate_test_set() -- for the
    households require_full_span excluded from this run -- and denormalize
    back into physical units (W).

    Returns (true, pred, user_ids, n_held_out) -- true/pred layouts match
    calculate_metrics.py's disaggregate_test_set() -- or None if this run
    isn't applicable (no checkpoint, node-locked model type, or no
    held-out users for this config).
    """
    if row["ckpt_path"] is None or row["model_type"] in NODE_LOCKED_MODEL_TYPES:
        return None

    held_out_ids = resolve_held_out_ids(row)  # also resets + reloads cfg
    if not held_out_ids:
        return None

    cfg.earne_data.filter_ids = held_out_ids
    cfg.earne_data.require_full_span = False
    # this script does one no-grad pass over an already-processed dataset;
    # background loader workers add nothing here and would multiply with
    # this script's own process-pool parallelism, oversubscribing the host
    cfg.num_workers = 0
    cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    # baseline_knn.py reads cfg.baseline.knn_cache_device independently of
    # cfg.accelerator (default "cuda") -- see calculate_metrics.py's
    # disaggregate_test_set() for the full explanation. baseline_knn is
    # already in NODE_LOCKED_MODEL_TYPES so it never reaches this function
    # today, but pin defensively in case that changes.
    cfg.baseline.knn_cache_device = cfg.accelerator
    device = torch.device(cfg.accelerator)
    targets = active_targets()
    n_q = cfg.model.n_quantiles

    # create_dataset() (not a hand-built EARNeGraphDataset) so
    # set_dataset_info() + the registered loader set cfg.share.dim_in/
    # dim_out/num_nodes and register.transform -- create_model() depends
    # on these being set first.
    dataset = create_dataset()
    _, val_idx, test_idx = dataset.get_split_indices()
    eval_idx = val_idx + test_idx  # same window as disaggregate_test_set()
    loader = PyGDataLoader(
        dataset[torch.tensor(eval_idx, dtype=torch.long)],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model()
    _load_state_dict(row["ckpt_path"], model)
    model = model.to(device).eval()

    true_chunks, pred_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, true = model(batch)  # (pred, true) per GraphGymModule.forward()

            t_per_batch = pred.shape[0] // dataset.num_nodes
            pred_chunks.append(
                pred.reshape(dataset.num_nodes, t_per_batch, n_q * len(targets))
                .permute(0, 2, 1)
                .cpu()
            )
            true_chunks.append(
                true.reshape(dataset.num_nodes, t_per_batch, 4)
                .permute(0, 2, 1)
                .cpu()
            )

    pred = torch.cat(pred_chunks, dim=2)
    true = torch.cat(true_chunks, dim=2)

    t = dataset.transform_obj  # same fitted Transform regardless of node subset
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

    return true_denorm, pred_denorm, dataset.active_ids, len(held_out_ids)


def compute_held_out_metrics(row: pd.Series) -> tuple[dict, list[dict], int] | None:
    """
    Disaggregate `row`'s held-out set once and score it two ways, exactly
    like calculate_metrics.py's compute_metrics(): pooled across all
    held-out households (-> held_out_metrics_long.csv) and per household
    (-> held_out_metrics_per_household.csv).
    """
    result = disaggregate_held_out_set(row)
    if result is None:
        return None
    true, pred, user_ids, n_held_out = result
    # cfg still reflects `row`'s config -- resolve_held_out_ids/
    # disaggregate_held_out_set's load_cfg call mutated the module-level
    # singleton and nothing has touched it since (this call happens
    # immediately after, synchronously).
    targets = active_targets()
    n_q = cfg.model.n_quantiles
    q50_idx = cfg.model.quantiles.index(0.5)

    pooled = DisaggregationMetrics.all(
        true.numpy(), pred.numpy(), targets=targets, n_quantiles=n_q, q50_idx=q50_idx,
    )

    strat = _strat_fields(row)
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
                        **strat,
                        "user_id": uid,
                        "target": target,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pooled, household_records, n_held_out


# ── parallel execution ──────────────────────────────────────────────────────
def _compute_held_out_metrics_worker(
    row: pd.Series, timeout_s: int = ROW_TIMEOUT_S,
) -> tuple[pd.Series, dict | None, list[dict] | None, int | None, str | None]:
    """Picklable ProcessPoolExecutor wrapper: returns (row, pooled,
    household_records, n_held_out, error) instead of raising, so the
    parent can preserve the existing FAILED-and-continue behavior."""
    try:
        result = with_row_timeout(compute_held_out_metrics, row, timeout_s)
    except Exception as e:
        return row, None, None, None, str(e)
    if result is None:
        return row, None, None, None, None
    pooled, household_records, n_held_out = result
    return row, pooled, household_records, n_held_out, None


def _metrics_to_records(row: pd.Series, metrics: dict, order: int) -> list[dict]:
    """Expand one run's compute_held_out_metrics() output into
    held_out_metrics_long.csv records -- mirrors calculate_metrics.py's
    helper of the same name."""
    strat = _strat_fields(row)
    return [
        {
            "sweep": row["sweep"],
            "run_name": row["run_name"],
            "seed": row["seed"],
            **strat,
            "target": target,
            "metric": metric,
            "value": value,
            "_order": order,
        }
        for target, target_metrics in metrics.items()
        for metric, value in target_metrics.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Process-pool size for parallel disaggregation (1 = legacy serial "
             "path, still auto-detects CUDA; >1 forces CPU-only workers to avoid "
             "GPU contention).",
    )
    parser.add_argument(
        "--row-timeout", type=int, default=ROW_TIMEOUT_S,
        help="Per-row wall-clock timeout in seconds (SIGALRM) — a row exceeding "
             "this is reported as TIMEOUT and skipped rather than blocking the batch.",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    manifest = discover_manifest()
    print(f"Discovered {len(manifest)} runs across {manifest['sweep'].nunique()} sweeps")

    runnable = manifest[
        manifest["ckpt_path"].notna()
        & ~manifest["model_type"].isin(NODE_LOCKED_MODEL_TYPES)
    ]
    skipped_no_ckpt = manifest["ckpt_path"].isna().sum()
    skipped_node_locked = manifest["model_type"].isin(NODE_LOCKED_MODEL_TYPES).sum()
    if skipped_no_ckpt:
        print(f"Skipping {skipped_no_ckpt} runs — no checkpoint")
    if skipped_node_locked:
        print(
            f"Skipping {skipped_node_locked} runs — node-locked model type "
            f"(can't score unseen households): {sorted(NODE_LOCKED_MODEL_TYPES)}"
        )

    records = []
    household_records = []
    no_held_out = 0

    if args.workers <= 1:
        for order, (_, row) in enumerate(
            tqdm.tqdm(runnable.iterrows(), total=len(runnable), desc="scoring held-out households")
        ):
            try:
                result = with_row_timeout(compute_held_out_metrics, row, args.row_timeout)
            except RowTimeout as e:
                print(f"TIMEOUT {row['sweep']}/{row['run_name']}/seed={row['seed']}: {e}")
                continue
            except Exception as e:
                print(f"FAILED  {row['sweep']}/{row['run_name']}/seed={row['seed']}: {e}")
                continue
            if result is None:
                no_held_out += 1
                continue
            pooled, hh_records, n_held_out = result
            print(f"{row['run_name']}: {n_held_out} held-out users")
            records.extend(_metrics_to_records(row, pooled, order))
            household_records.extend(hh_records)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx, initializer=_init_worker,
            # see calculate_metrics.py's matching ProcessPoolExecutor call
            # for why: a worker that hits WORKER_MEM_LIMIT_GB stays pinned
            # near its cap and poisons every later row it handles, so
            # retire each worker after one task.
            max_tasks_per_child=1,
        ) as executor:
            future_to_order = {
                executor.submit(_compute_held_out_metrics_worker, row, args.row_timeout): order
                for order, (_, row) in enumerate(runnable.iterrows())
            }
            for future in tqdm.tqdm(
                as_completed(future_to_order), total=len(future_to_order),
                desc="scoring held-out households",
            ):
                row, pooled, hh_records, n_held_out, error = future.result()
                if error is not None:
                    tag = "TIMEOUT" if "row exceeded" in error else "FAILED "
                    print(f"{tag} {row['sweep']}/{row['run_name']}/seed={row['seed']}: {error}")
                    continue
                if pooled is None:
                    no_held_out += 1
                    continue
                print(f"{row['run_name']}: {n_held_out} held-out users")
                records.extend(_metrics_to_records(row, pooled, future_to_order[future]))
                household_records.extend(hh_records)

    if no_held_out:
        print(f"Skipping {no_held_out} runs — no held-out users for that config")

    df_long = pd.DataFrame(records)
    if df_long.empty:
        print("No runs were scored against held-out households — nothing to report.")
    else:
        df_long = df_long.sort_values(["_order", "target", "metric"]).drop(columns="_order")
        df_long.to_csv("notebooks/held_out_metrics_long.csv", index=False)

        df = df_long.pivot_table(
            index="metric",
            columns=["target", "sweep", "run_name"],
            values="value",
            aggfunc="mean",
        )
        df = prune_metrics(df)
        df = df.rename(index=METRIC_LABELS)
        df.to_csv("notebooks/held_out_metrics_wide.csv")
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)
        print(df)

    df_household = pd.DataFrame(household_records)
    if not df_household.empty:
        df_household = df_household.sort_values(
            ["sweep", "run_name", "seed", "user_id", "target", "metric"]
        )
        df_household.to_csv("notebooks/held_out_metrics_per_household.csv", index=False)
        print(
            f"[OK] wrote notebooks/held_out_metrics_per_household.csv ({len(df_household)} rows, "
            f"{df_household['run_name'].nunique()} runs, {df_household['user_id'].nunique()} households)"
        )
