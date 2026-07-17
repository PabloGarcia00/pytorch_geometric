from pathlib import Path
from typing import Dict

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric

from ..target_utils import active_targets
from ..transform.transform import Transform

_TRANSFORM_CACHE = {}


def _get_transform() -> Transform:
    mode = "dual" if cfg.model.dim_in == 2 else "single"
    # earne_loader_new reads/writes under earne_data.processed_root, not
    # dataset.dir (which goes unused and can point somewhere else entirely).
    dataset_dir = cfg.earne_data.processed_root
    cache_key = f"{dataset_dir}_{mode}"

    if cache_key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[cache_key]

    t = Transform.load(Path(dataset_dir) / f"transform_{mode}.pt")
    _TRANSFORM_CACHE[cache_key] = t
    return t


@torch.no_grad()
def compute_st_caps_errors(true_list, pred_list) -> Dict[str, float]:
    """
    true_list: list of tensors [N, 4] (y_load, y_pv, mask, y_net_demand) — scaled space
    pred_list: list of tensors [N, len(targets)] (one column per
        cfg.model.predict_targets entry, in active_targets() order)      — scaled space

    Denormalizes the selected stream(s) back into watts before scoring,
    masked to timesteps that actually have a label (mirrors compute_earne_mae).
    Averages "mae"/"rmse" only over whichever target(s) were predicted.
    """
    t = _get_transform()
    targets = active_targets()
    true_col = {"load": 0, "pv": 1}

    totals = {name: {"abs": 0.0, "sq": 0.0, "n": 0.0} for name in targets}

    for true, pred in zip(true_list, pred_list):
        mask = true[:, 2]
        for i, name in enumerate(targets):
            err = t.inverse_transform(name, pred[:, i]) - t.inverse_transform(
                name, true[:, true_col[name]]
            )
            totals[name]["abs"] += (err.abs() * mask).sum().item()
            totals[name]["sq"] += ((err**2) * mask).sum().item()
            totals[name]["n"] += mask.sum().item()

    if any(totals[name]["n"] == 0 for name in targets):
        return {"mae": 0.0, "rmse": 0.0}

    maes = [totals[name]["abs"] / totals[name]["n"] for name in targets]
    rmses = [(totals[name]["sq"] / totals[name]["n"]) ** 0.5 for name in targets]

    return {
        "mae": sum(maes) / len(maes),
        "rmse": sum(rmses) / len(rmses),
    }


@register_metric('st_caps_rmse')
def st_caps_rmse(true_list, pred_list, task_type):
    rmse = compute_st_caps_errors(true_list, pred_list)["rmse"]
    return float(round(rmse, 3))


@register_metric('st_caps_mae')
def st_caps_mae(true_list, pred_list, task_type):
    mae = compute_st_caps_errors(true_list, pred_list)["mae"]
    return float(round(mae, 3))
