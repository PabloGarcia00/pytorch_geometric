import torch
import torch.nn.functional as F
from typing import Dict

from torch_geometric.graphgym.register import register_metric
from torch_geometric.graphgym.config import cfg

@register_metric('st_caps_rmse')
def st_caps_rmse(true_list, pred_list, task_type):
    # true_list: list of tensors [N, 3] (y_load, y_pv, mask)
    # pred_list: list of tensors [N, 2] (load_pred, pv_pred)
    all_true = torch.cat(true_list)
    all_pred = torch.cat(pred_list)
    
    # Calculate RMSE for both Load and PV
    metrics_load = calculate_metrics(all_pred[:, 0], all_true[:, 0])
    metrics_pv = calculate_metrics(all_pred[:, 1], all_true[:, 1])
    
    # Return average RMSE for simplicity or choose one
    return (metrics_load['rmse'] + metrics_pv['rmse']) / 2.0

@register_metric('st_caps_mae')
def st_caps_mae(true_list, pred_list, task_type):
    all_true = torch.cat(true_list)
    all_pred = torch.cat(pred_list)
    metrics_load = calculate_metrics(all_pred[:, 0], all_true[:, 0])
    metrics_pv = calculate_metrics(all_pred[:, 1], all_true[:, 1])
    return (metrics_load['mae'] + metrics_pv['mae']) / 2.0

def calculate_metrics(pred: torch.Tensor, true: torch.Tensor,
                      denorm_mean: float = 0, denorm_std: float = 1) -> Dict:
    pred = pred * denorm_std + denorm_mean
    true = true * denorm_std + denorm_mean
    true = torch.abs(true) + 1e-6
    rmse = torch.sqrt(F.mse_loss(pred, true))
    mae = F.l1_loss(pred, true)
    mape = torch.mean(torch.abs((true - pred) / true)) * 100
    return {
        'rmse': rmse.item(),
        'mae':  mae.item(),
        'mape': mape.item()
    }

def aggregate_metrics(all_preds: torch.Tensor, all_true: torch.Tensor,
                      prefix: str = '') -> Dict:
    metrics = calculate_metrics(all_preds, all_true)
    return {
        f'{prefix}rmse': metrics['rmse'],
        f'{prefix}mae':  metrics['mae'],
        f'{prefix}mape': metrics['mape']
    }
