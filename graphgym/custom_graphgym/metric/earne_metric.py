import torch
import joblib
from pathlib import Path
from torch_geometric.graphgym.register import register_metric
from torch_geometric.graphgym.config import cfg

def _get_denorm_scale():
    """
    Returns a scalar (in original Watt units) to convert normalised MAE back to physical units.
    Supports both minmax (scaler.pkl) and zero_log (norm_params.pt) modes.
    """
    norm_mode = cfg.earne_data.get('norm_mode', 'minmax')
    dataset_dir = cfg.dataset.dir

    if norm_mode == 'minmax':
        scaler = joblib.load(Path(dataset_dir) / 'scaler.pkl')
        return float(scaler.data_range_[0])

    elif norm_mode == 'zero_log':
        params = torch.load(Path(dataset_dir) / 'norm_params.pt', weights_only=False)
        # nz_std is in log-space; exp(nz_std) gives a rough Watt-scale denominator
        return float(torch.exp(params['nz_std']).squeeze())

    raise ValueError(f"Unknown norm_mode: {norm_mode}")

def compute_earne_mae(true_list, pred_list):
    """
    true_list: A list of tensors [N, 4] (y_load, y_pv, mask, y_net_demand)
    pred_list: A list of tensors [N, 2 * n_quantiles]
    """
    norm_mode = cfg.earne_data.get('norm_mode', 'minmax')
    dataset_dir = cfg.dataset.dir
    
    if norm_mode == 'minmax':
        scaler = joblib.load(Path(dataset_dir) / 'scaler.pkl')
        scale = float(scaler.data_range_[0])
        min_val = float(scaler.data_min_[0])
        def denorm(x): return x * scale + min_val
    elif norm_mode == 'zero_log':
        from ..loader.graph_dataset import zero_preserved_log_denormalize
        params = torch.load(Path(dataset_dir) / 'norm_params.pt', weights_only=False)
        nz_mean = params['nz_mean']
        nz_std = params['nz_std']
        def denorm(x): return zero_preserved_log_denormalize(x, nz_mean, nz_std)
    
    median_idx = cfg.model.quantiles.index(0.5)

    total_mae_load = 0.0
    total_mae_pv = 0.0
    total_nodes = 0

    for true, pred in zip(true_list, pred_list):
        y_load = true[:, 0]
        y_pv = true[:, 1]
        mask = true[:, 2]

        q_load_median = pred[:, median_idx]
        q_pv_median = pred[:, cfg.model.n_quantiles + median_idx]
        
        # Denormalize to physical units (Watts) before MAE
        err_load = torch.abs(denorm(q_load_median) - denorm(y_load))
        err_pv = torch.abs(denorm(q_pv_median) - denorm(y_pv))
        
        total_mae_load += (err_load * mask).sum().item()
        total_mae_pv += (err_pv * mask).sum().item()
        total_nodes += mask.sum().item()

    if total_nodes == 0:
        return 0.0, 0.0
        
    final_mae_load = total_mae_load / total_nodes
    final_mae_pv = total_mae_pv / total_nodes
    return final_mae_load, final_mae_pv

@register_metric('earne_mae_load')
def earne_mae_load(true_list, pred_list, task_type):
    mae_load, _ = compute_earne_mae(true_list, pred_list)
    return float(mae_load)

@register_metric('earne_mae_pv')
def earne_mae_pv(true_list, pred_list, task_type):
    _, mae_pv = compute_earne_mae(true_list, pred_list)
    return float(mae_pv)
