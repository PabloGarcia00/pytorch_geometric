import argparse
import fnmatch
import json
import multiprocessing
import os
import re
import resource
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# polars (used by custom_graphgym's loader) defaults its internal async
# I/O thread pool to os.cpu_count() with no cap -- fine for a single
# process, but with N ProcessPoolExecutor workers each independently
# grabbing up to nproc threads, the host's async I/O resources (io_uring/
# epoll) get oversubscribed and polars' Rust runtime panics with EAGAIN
# ("Resource temporarily unavailable"), killing that worker and poisoning
# the whole pool via BrokenProcessPool. Must be set before polars' first
# import (including the re-exec of this module's top-level imports in
# each spawned child under the "spawn" start method), so this has to be
# the first thing this file does.
os.environ.setdefault("POLARS_MAX_THREADS", "2")

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
from torch_geometric.graphgym.loader import create_dataset
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.loader import DataLoader as PyGDataLoader

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



# Flat, one-level results_root/<run>/config.yaml families (no separate sweep
# directory, unlike the <sweep>/<run>/ grid-sweep runs) -- each gets a
# synthetic sweep label since there's no real sweep directory to read one
# from. Matched by glob pattern below; first matching prefix/suffix wins.
_FLAT_SWEEP_GLOBS = {
    "coverage_eval_*": "coverage_eval",  # require_full_span True/False ablation
    "*_pw03": "physics_weight_nomask",  # mask_physics_impossible=False + physics_weight=0.3
    "res_eval_*": "resolution_eval",  # 5/30/60-min reprocessed-resolution ablation
    "trtr": "tstr_trts_eval",  # exact match — literal dir "results/trtr"
    "trts_*": "tstr_trts_eval",  # results/trts_{energydiff,faraday,guide_vae}
    "tstr_*": "tstr_trts_eval",  # results/tstr_{energydiff,faraday,guide_vae} +
    # results/tstr_trts_<model>_* grid (all start with "tstr_", so this one
    # glob covers both without double-counting — disjoint from "trts_*")
}


def _flat_sweep_label(run_dir: Path) -> str:
    for pattern, label in _FLAT_SWEEP_GLOBS.items():
        if run_dir.match(pattern):
            return label
    return run_dir.name  # fallback -- shouldn't hit given the globs below


def _parse_resolution(processed_root: str | None) -> str:
    """'datasets/earne_csi_5min' -> '5min'; 'datasets/earne_csi' (no
    suffix, the default reprocessing cadence) -> 'native'. Resolution
    ablation configs (res_eval_*) are the only ones with this suffix --
    everything else, including the tstr_trts_* families whose
    processed_root ends in a model/protocol name rather than a digit, are
    unaffected."""
    m = re.search(r"_(\d+min)$", processed_root or "")
    return m.group(1) if m else "native"


def _parse_tstr_trts(gold_data: str | None) -> tuple[str | None, str | None]:
    """Parse the TSTR/TRTS train/test-data protocol and (if any) synthetic
    generator baseline out of gold_data's filename -- e.g.
    '.../trts_energydiff_gold.parquet' -> ('trts', 'energydiff'),
    '.../trtr_gold.parquet' -> ('trtr', None) (real-to-real has no
    synthetic source). Returns (None, None) for every non-TSTR/TRTS config
    (gold_data pointing at the real fleet_gold_layer*.parquet)."""
    stem = Path(gold_data or "").stem
    match = re.match(r"^(trtr|trts|tstr)(?:_(\w+))?_gold$", stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def discover_manifest(results_root: Path | str = RESULTS_ROOT) -> pd.DataFrame:
    """
    Walk `results_root/<sweep>/<run>/<seed>/` and return one row per (sweep, run, seed)
    pointing at that replicate's config, checkpoint, and stats — the starting point for
    running/aggregating metrics generically across every disaggregation model in `results/`.

    Also picks up the flat one-level results_root/<run>/config.yaml families in
    _FLAT_SWEEP_GLOBS (coverage_eval_* require_full_span ablation, *_pw03
    no-mask+physics-loss runs, res_eval_* resolution ablation, trtr/trts_*/
    tstr_* TSTR/TRTS synthetic-data protocol) -- these don't nest under a
    separate sweep directory the way the grid-sweep runs above do, so they
    need their own glob(s) and a synthetic sweep label (see
    _flat_sweep_label()).

    Beyond the raw config fields, four factors are parsed out into their
    own columns because they aren't otherwise crisply queryable:
    node_encoder_name (the real GNN/LSTM/MLP discriminator within the
    shared model_type == 'earne_network'), physics_weight (numeric soft
    penalty, distinct from the mask_physics_impossible hard toggle),
    resolution (5min/30min/60min/native, parsed from processed_root), and
    tstr_protocol/synthetic_source (parsed from gold_data's filename).
    """
    results_root = Path(results_root)
    records = []

    config_paths = sorted(results_root.glob("*/*/config.yaml"))
    for pattern in _FLAT_SWEEP_GLOBS:
        config_paths += sorted(results_root.glob(f"{pattern}/config.yaml"))

    for config_path in config_paths:
        run_dir = config_path.parent
        cfg_yaml = yaml.safe_load(config_path.read_text())
        dataset_cfg = cfg_yaml.get("dataset", {})
        earne_cfg = cfg_yaml.get("earne_data", {})
        model_cfg = cfg_yaml.get("model", {})
        train_cfg = cfg_yaml.get("train", {})
        tstr_protocol, synthetic_source = _parse_tstr_trts(earne_cfg.get("gold_data"))

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
                    "sweep": (
                        _flat_sweep_label(run_dir)
                        if run_dir.parent == results_root
                        else run_dir.parent.name
                    ),
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
                    "resolution": _parse_resolution(earne_cfg.get("processed_root")),
                    "tstr_protocol": tstr_protocol,  # None / 'trtr' / 'trts' / 'tstr'
                    "synthetic_source": synthetic_source,  # None / 'energydiff' / 'faraday' / 'guide_vae'
                    "model_type": model_cfg.get("type"),
                    # model_type alone conflates GNN/LSTM/MLP under the
                    # shared 'earne_network' wrapper -- node_encoder_name is
                    # the actual architecture discriminator between them
                    # (earne_temporal / baseline_lstm_temporal /
                    # baseline_mlp_temporal). Every other model_type just
                    # uses earne_temporal regardless, so this column is
                    # only informative when model_type == 'earne_network'.
                    "node_encoder_name": model_cfg.get("node_encoder_name"),
                    "physics_weight": train_cfg.get("physics_weight"),
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


# Columns from a discover_manifest() row that are threaded straight into
# every metrics_long.csv / metrics_per_household.csv record (and
# held_out_metrics.py's equivalents), so a result table can be stratified
# on any of these without a manual join back to the manifest.
STRAT_COLUMNS = [
    "model_type",
    "node_encoder_name",  # only informative when model_type == 'earne_network' (GNN/LSTM/MLP)
    "dual_read",
    "weather_mode",
    "graph_mode",
    "mask_physics_impossible",
    "physics_weight",
    "require_full_span",
    "resolution",
    "tstr_protocol",
    "synthetic_source",
]


def _strat_fields(row: pd.Series) -> dict:
    return {col: row[col] for col in STRAT_COLUMNS}


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
    Load `row`'s config + checkpoint, run inference over its val+test
    window, and denormalize predictions/labels back into physical units
    (W).

    Scores val+test COMBINED, not test alone: on this dataset's 3-year
    span, a chronological 70/15/15 split puts the tail 15% ("test") in
    late-Oct-through-early-Apr -- entirely autumn/winter/early-spring, no
    full summer -- while "val" (mid-May through late-Oct) does cover a
    full summer. val is never used for gradient updates (only best-val-
    loss checkpoint selection), so folding it into the reported eval
    window is leak-safe and gives a season-balanced number instead of a
    winter-only one. See held_out_metrics.py's disaggregate_held_out_set()
    for the matching window on the held-out population.

    Returns (true, pred, user_ids, timestamps):
        true: [N, 4, T]  — y_load, y_pv, mask, y_net_demand (always both,
            regardless of what was predicted)
        pred: [N, n_quantiles * len(targets), T] — one quantile block per
            cfg.model.predict_targets entry, in active_targets() order
        user_ids: length-N list, aligned with true/pred's node dimension
        timestamps: length-T list of the real timestamp each T column
            corresponds to, in ascending val-then-test order (_calc_splits
            returns each split as a contiguous, ascending index range).
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
    # baseline_knn.py reads cfg.baseline.knn_cache_device independently of
    # cfg.accelerator (default "cuda" — a training-time optimization baked
    # into every baseline_knn config), so it ignores _init_worker()'s
    # CPU-only masking unless pinned here too. Left unpinned, concurrent
    # baseline_knn workers all reach for the same physical GPU and OOM it,
    # taking down the whole ProcessPoolExecutor via BrokenProcessPool.
    cfg.baseline.knn_cache_device = cfg.accelerator
    device = torch.device(cfg.accelerator)
    targets = active_targets()
    n_q = cfg.model.n_quantiles

    # create_dataset() (not GraphGymDataModule) so we can build our own
    # val+test loader instead of being confined to test_graph_index --
    # set_dataset_info() + the registered loader still set
    # cfg.share.dim_in/dim_out/num_nodes and register.transform, exactly
    # as GraphGymDataModule()'s internal create_loader() would.
    dataset = create_dataset()
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    eval_idx = val_idx + test_idx  # contiguous, ascending (val precedes test)

    eval_loader = PyGDataLoader(
        dataset[torch.tensor(eval_idx, dtype=torch.long)],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model()
    if cache_fit:
        train_loader = PyGDataLoader(
            dataset[torch.tensor(train_idx, dtype=torch.long)],
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        model.model.fit_cache(train_loader)
    else:
        _load_state_dict(row["ckpt_path"], model)
    model = model.to(device).eval()

    true_chunks, pred_chunks = [], []
    with torch.no_grad():
        for batch in eval_loader:
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
    timestamps = [dataset.timestamps[i + dataset.seq_len] for i in eval_idx]

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

    return pooled, household_records


# ── per-row timeout / memory guard ──────────────────────────────────────────
# Two independent safeguards against one pathological row taking down the
# whole batch (history: a stuck row with a 73-100GB RSS anomaly ran for
# 1.75+ days; killing its worker mid-pool then poisoned every other pending
# future with BrokenProcessPool, losing all in-memory output). Neither
# safeguard kills the worker process itself — both raise a catchable
# exception inside that one row's call stack, so the worker just reports a
# TIMEOUT/MemoryError for that row and moves on to its next queued task.
ROW_TIMEOUT_S = 10800  # 3h — generous vs. observed legit-slow 5min-res rows
                        # (~45min worst case seen so far), bounds the rest
WORKER_MEM_LIMIT_GB = 64  # RLIMIT_AS caps *virtual* address space, not RSS.
                            # Merely importing torch_geometric in a spawned
                            # worker (before this initializer even runs)
                            # reserves ~36-38GB of VAS from CUDA driver/UVA
                            # setup across this host's 8 visible GPUs --
                            # confirmed via /proc/self/maps, unrelated to
                            # actual row size. A 24GB cap sat *below* that
                            # unavoidable floor, so every worker started
                            # already over budget and any next allocation
                            # (e.g. mid yaml.safe_load, or mid torch.load's
                            # zip central-directory read) threw a spurious
                            # MemoryError that looked like file corruption.
                            # 64GB clears the ~38GB floor with headroom for
                            # real row work while staying well under the
                            # observed 73-100GB pathological-row RSS case.


class RowTimeout(Exception):
    pass


def with_row_timeout(fn, row: pd.Series, timeout_s: int = ROW_TIMEOUT_S):
    """Run fn(row), aborting with RowTimeout after timeout_s (SIGALRM).
    Each ProcessPoolExecutor worker processes one row at a time in its own
    process, so the alarm can't affect other in-flight rows in other
    workers."""
    def _handler(signum, frame):
        raise RowTimeout(f"row exceeded {timeout_s}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        return fn(row)
    finally:
        signal.alarm(0)


# ── parallel execution ──────────────────────────────────────────────────────
def _compute_metrics_worker(
    row: pd.Series, timeout_s: int = ROW_TIMEOUT_S,
) -> tuple[pd.Series, dict | None, list[dict] | None, str | None]:
    """Picklable ProcessPoolExecutor wrapper: returns (row, pooled, household_records,
    error) instead of raising, so the parent can preserve the existing
    FAILED-and-continue behavior."""
    try:
        result = with_row_timeout(compute_metrics, row, timeout_s)
    except Exception as e:
        return row, None, None, str(e)
    if result is None:
        return row, None, None, None
    pooled, household_records = result
    return row, pooled, household_records, None


def _init_worker() -> None:
    """Runs once per spawned worker before any task: forces CPU-only inference
    (avoids GPU contention/OOM across workers sharing the same GPU(s)), caps
    intra-op/BLAS threading so `--workers` processes don't each try to claim
    every core on the host, and caps address space so one pathological row
    (see ROW_TIMEOUT_S/WORKER_MEM_LIMIT_GB above) can't balloon this worker's
    RSS and starve the rest of the host."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    limit_bytes = WORKER_MEM_LIMIT_GB * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, resource.RLIM_INFINITY))


def _metrics_to_records(row: pd.Series, metrics: dict, order: int) -> list[dict]:
    """Expand one row's compute_metrics() output into metrics_long.csv records —
    shared by both the serial and parallel __main__ branches."""
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
    parser.add_argument(
        "--row-timeout", type=int, default=ROW_TIMEOUT_S,
        help="Per-row wall-clock timeout in seconds (SIGALRM) — a row exceeding "
             "this is reported as TIMEOUT and skipped rather than blocking the batch.",
    )
    parser.add_argument(
        "--run-name-glob", type=str, default=None,
        help="Only score runs whose run_name matches this glob (fnmatch syntax, "
             "e.g. 'res_eval_*_5min_dualmask') — lets a small, slow-model batch "
             "be scored separately from the bulk sweep, e.g. with --workers 1 "
             "so disaggregate_test_set()'s cfg.accelerator=auto picks up a real "
             "GPU (only the >1-worker path forces CPU via _init_worker, to avoid "
             "many workers contending for one GPU) without paying serial-mode's "
             "cost for the hundreds of other, already-fast-on-CPU runs. Default: "
             "no filter, scores everything (unchanged behavior).",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    manifest = discover_manifest()
    print(f"Discovered {len(manifest)} runs across {manifest['sweep'].nunique()} sweeps")

    if args.run_name_glob:
        manifest = manifest[
            manifest["run_name"].apply(
                lambda n: fnmatch.fnmatch(n, args.run_name_glob)
            )
        ]
        print(
            f"--run-name-glob {args.run_name_glob!r}: {len(manifest)} runs match"
        )

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
                result = with_row_timeout(compute_metrics, row, args.row_timeout)
            except RowTimeout as e:
                print(f"TIMEOUT {row['sweep']}/{row['run_name']}/seed={row['seed']}: {e}")
                continue
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
            max_workers=args.workers, mp_context=ctx, initializer=_init_worker,
            # A row that hits WORKER_MEM_LIMIT_GB is caught cleanly (see
            # with_row_timeout/_init_worker above), but the worker process
            # doesn't release that claimed virtual address space afterward
            # -- it stays pinned near its RLIMIT_AS ceiling and every
            # subsequent row it picks up fails too, eventually killing the
            # worker outright and poisoning the whole pool
            # (BrokenProcessPool). Retiring each worker after one task gives
            # every row a fresh process/fresh memory ceiling.
            max_tasks_per_child=1,
        ) as executor:
            future_to_order = {
                executor.submit(_compute_metrics_worker, row, args.row_timeout): order
                for order, (_, row) in enumerate(runnable.iterrows())
            }
            for future in tqdm.tqdm(
                as_completed(future_to_order), total=len(future_to_order),
                desc="disaggregating test sets",
            ):
                row, pooled, hh_records, error = future.result()
                if error is not None:
                    tag = "TIMEOUT" if "row exceeded" in error else "FAILED "
                    print(f"{tag} {row['sweep']}/{row['run_name']}/seed={row['seed']}: {error}")
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
