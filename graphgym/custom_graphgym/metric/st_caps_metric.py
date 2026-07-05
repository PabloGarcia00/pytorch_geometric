import torch
import torch.nn.functional as F
from typing import Dict

from torch_geometric.graphgym.register import register_metric


def calculate_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict:
    true = torch.abs(true) + 1e-6
    rmse = torch.sqrt(F.mse_loss(pred, true))
    mae = F.l1_loss(pred, true)
    mape = torch.mean(torch.abs((true - pred) / true)) * 100
    return {'rmse': rmse.item(), 'mae': mae.item(), 'mape': mape.item()}


@register_metric('st_caps_rmse')
def st_caps_rmse(true_list, pred_list, task_type):
    all_true = torch.cat(true_list)
    all_pred = torch.cat(pred_list)
    m_load = calculate_metrics(all_pred[:, 0], all_true[:, 0])
    m_pv = calculate_metrics(all_pred[:, 1], all_true[:, 1])
    return (m_load['rmse'] + m_pv['rmse']) / 2.0


@register_metric('st_caps_mae')
def st_caps_mae(true_list, pred_list, task_type):
    all_true = torch.cat(true_list)
    all_pred = torch.cat(pred_list)
    m_load = calculate_metrics(all_pred[:, 0], all_true[:, 0])
    m_pv = calculate_metrics(all_pred[:, 1], all_true[:, 1])
    return (m_load['mae'] + m_pv['mae']) / 2.0
