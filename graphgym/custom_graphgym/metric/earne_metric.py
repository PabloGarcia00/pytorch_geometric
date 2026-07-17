import math
from pathlib import Path

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
def compute_earne_mae(true_list, pred_list):
    """
    true_list: list of tensors [N, 4] (y_load, y_pv, mask, y_net_demand)
    pred_list: list of tensors [N, n_quantiles * len(targets)], one
        quantile block per cfg.model.predict_targets entry (default: PV
        only), in active_targets() order.

    Returns (mae_load, mae_pv); either is NaN if that target isn't in
    cfg.model.predict_targets (there's no corresponding pred column block
    to compute it from).
    """
    t = _get_transform()
    median_idx = cfg.model.quantiles.index(0.5)
    n_q = cfg.model.n_quantiles
    targets = active_targets()

    totals = {name: {"abs": 0.0, "n": 0.0} for name in targets}
    y_col = {"load": 0, "pv": 1}

    for true, pred in zip(true_list, pred_list):
        mask = true[:, 2]
        for i, name in enumerate(targets):
            y = true[:, y_col[name]]
            q_median = torch.nan_to_num(pred[:, i * n_q + median_idx], nan=0.0)
            err = torch.abs(
                t.inverse_transform(name, q_median) - t.inverse_transform(name, y)
            )
            totals[name]["abs"] += (err * mask).sum().item()
            totals[name]["n"] += mask.sum().item()

    def _mae(name):
        if name not in totals or totals[name]["n"] == 0:
            return float("nan")
        return totals[name]["abs"] / totals[name]["n"]

    return _mae("load"), _mae("pv")


@register_metric("earne_mae_load")
def earne_mae_load(true_list, pred_list, task_type):
    mae_load, _ = compute_earne_mae(true_list, pred_list)
    return float("nan") if math.isnan(mae_load) else float(round(mae_load, 3))


@register_metric("earne_mae_pv")
def earne_mae_pv(true_list, pred_list, task_type):
    _, mae_pv = compute_earne_mae(true_list, pred_list)
    return float("nan") if math.isnan(mae_pv) else float(round(mae_pv, 3))
