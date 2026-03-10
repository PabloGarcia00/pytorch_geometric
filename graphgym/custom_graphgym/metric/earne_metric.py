import torch
import joblib
from torch_geometric.graphgym.register import register_metric
from torch_geometric.graphgym.config import cfg

def compute_earne_mae(true_list, pred_list):
    """
    true_list: A list of tensors [N, 3] (y_load, y_pv, mask)
    pred_list: A list of tensors [N, 2 * n_quantiles]
    """
    scaler_path = f"{cfg.dataset.dir}/scaler.pkl"
    scaler = joblib.load(scaler_path)
    scale = scaler.data_range_[0]
    
    total_mae_load = 0.0
    total_mae_pv = 0.0
    total_nodes = 0

    for true, pred in zip(true_list, pred_list):
        y_load = true[:, 0]
        y_pv = true[:, 1]
        mask = true[:, 2]
        
        q_load_median = pred[:, 1]
        q_pv_median = pred[:, cfg.model.n_quantiles + 1]
        
        mae_load = (torch.abs(q_load_median - y_load) * mask).sum()
        mae_pv = (torch.abs(q_pv_median - y_pv) * mask).sum()
        
        total_mae_load += mae_load.item()
        total_mae_pv += mae_pv.item()
        total_nodes += mask.sum().item()

    if total_nodes == 0:
        return 0.0, 0.0
        
    final_mae_load = (total_mae_load / total_nodes) * scale
    final_mae_pv = (total_mae_pv / total_nodes) * scale
    return final_mae_load, final_mae_pv

@register_metric('earne_mae_load')
def earne_mae_load(true_list, pred_list, task_type):
    mae_load, _ = compute_earne_mae(true_list, pred_list)
    return float(mae_load)

@register_metric('earne_mae_pv')
def earne_mae_pv(true_list, pred_list, task_type):
    _, mae_pv = compute_earne_mae(true_list, pred_list)
    return float(mae_pv)
