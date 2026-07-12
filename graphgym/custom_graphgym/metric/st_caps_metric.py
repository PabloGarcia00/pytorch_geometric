from pathlib import Path
from typing import Dict

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric

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
    pred_list: list of tensors [N, 2]  (load_pred, pv_pred)               — scaled space

    Denormalizes both streams back into watts before scoring, masked to
    timesteps that actually have a label (mirrors compute_earne_mae).
    """
    t = _get_transform()

    total_abs_load = total_sq_load = 0.0
    total_abs_pv = total_sq_pv = 0.0
    total_load_nodes = total_pv_nodes = 0.0

    for true, pred in zip(true_list, pred_list):
        mask = true[:, 2]

        err_load = t.inverse_transform("load", pred[:, 0]) - t.inverse_transform(
            "load", true[:, 0]
        )
        err_pv = t.inverse_transform("pv", pred[:, 1]) - t.inverse_transform(
            "pv", true[:, 1]
        )

        total_abs_load += (err_load.abs() * mask).sum().item()
        total_sq_load += ((err_load**2) * mask).sum().item()
        total_abs_pv += (err_pv.abs() * mask).sum().item()
        total_sq_pv += ((err_pv**2) * mask).sum().item()
        total_load_nodes += mask.sum().item()
        total_pv_nodes += mask.sum().item()

    if total_load_nodes == 0:
        return {"mae": 0.0, "rmse": 0.0}

    mae_load = total_abs_load / total_load_nodes
    mae_pv = total_abs_pv / total_pv_nodes
    rmse_load = (total_sq_load / total_load_nodes) ** 0.5
    rmse_pv = (total_sq_pv / total_pv_nodes) ** 0.5

    return {
        "mae": (mae_load + mae_pv) / 2.0,
        "rmse": (rmse_load + rmse_pv) / 2.0,
    }


@register_metric('st_caps_rmse')
def st_caps_rmse(true_list, pred_list, task_type):
    rmse = compute_st_caps_errors(true_list, pred_list)["rmse"]
    return float(round(rmse, 3))


@register_metric('st_caps_mae')
def st_caps_mae(true_list, pred_list, task_type):
    mae = compute_st_caps_errors(true_list, pred_list)["mae"]
    return float(round(mae, 3))
