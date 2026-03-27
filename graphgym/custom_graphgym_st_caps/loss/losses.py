import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register

@register_loss('st_caps_loss')
def st_caps_loss_complex(pred, true):
    if cfg.model.loss_fun != 'st_caps_loss':
        return None
    
    # Access intermediate outputs via registered batch
    batch = getattr(register, 'batch', None)
    if batch is None or not hasattr(batch, 'st_caps_outputs'):
        return torch.tensor(0.0, requires_grad=True, device=pred.device), pred
        
    outputs = batch.st_caps_outputs
    
    # Instantiate or reuse the multi-component loss
    loss_module = STSGCCapsLoss(
        lambda_S=cfg.st_caps.lambda_S,
        lambda_L=cfg.st_caps.lambda_L,
        lambda_PV=cfg.st_caps.lambda_PV,
        lambda_SC=cfg.st_caps.lambda_SC
    )
    
    total_loss, metrics = loss_module(outputs)
    
    # Store metrics for logging if possible
    # GraphGym loggers pull from loss but metrics are trickier
    
    return total_loss, pred

class ReconstructionLoss(nn.Module):
    def __init__(self, lambda_S: float = 1.0):
        super().__init__()
        self.lambda_S = lambda_S

    def forward(self, edge_pred: torch.Tensor, edge_true: torch.Tensor,
                node_pred: torch.Tensor, node_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        j_E = F.mse_loss(edge_pred, edge_true)
        j_S = F.mse_loss(node_pred, node_true)
        total_loss = j_E + self.lambda_S * j_S
        return total_loss, {
            'edge_loss':  j_E.item(),
            'node_loss':  j_S.item(),
            'recon_loss': total_loss.item()
        }

class SparseCodingLoss(nn.Module):
    def __init__(self, lambda_SC: float = 0.1):
        super().__init__()
        self.lambda_SC = lambda_SC

    def forward(self, features: torch.Tensor, reconstruction: torch.Tensor,
                codes: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        recon_error = F.mse_loss(reconstruction, features)
        l1_penalty = torch.mean(torch.abs(codes))
        total_loss = recon_error + self.lambda_SC * l1_penalty
        return total_loss, {
            'sc_recon_error': recon_error.item(),
            'sc_l1_penalty':  l1_penalty.item(),
            'sc_loss':        total_loss.item()
        }

class DisaggregationLoss(nn.Module):
    def __init__(self, lambda_L: float = 1.0, lambda_PV: float = 1.0):
        super().__init__()
        self.lambda_L = lambda_L
        self.lambda_PV = lambda_PV

    def forward(self, load_pred: torch.Tensor, load_true: torch.Tensor,
                pv_pred: torch.Tensor, pv_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        j_L = F.mse_loss(load_pred, load_true)
        j_PV = F.mse_loss(pv_pred, pv_true)
        total_loss = self.lambda_L * j_L + self.lambda_PV * j_PV
        return total_loss, {
            'load_loss': j_L.item(),
            'pv_loss':   j_PV.item(),
            'est_loss':  total_loss.item()
        }

class STSGCCapsLoss(nn.Module):
    def __init__(self, lambda_S: float = 1.0, lambda_L: float = 1.0,
                 lambda_PV: float = 1.0, lambda_SC: float = 0.1):
        super().__init__()
        self.recon_loss = ReconstructionLoss(lambda_S)
        self.sc_loss = SparseCodingLoss(lambda_SC)
        self.disagg_loss = DisaggregationLoss(lambda_L, lambda_PV)

    def forward(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        metrics = {}
        total_loss = 0
        if 'edge_pred' in outputs and 'edge_true' in outputs:
            recon_loss, recon_metrics = self.recon_loss(
                outputs['edge_pred'], outputs['edge_true'],
                outputs['node_pred'], outputs['node_true']
            )
            total_loss = total_loss + recon_loss
            metrics.update(recon_metrics)
        if 'features' in outputs and 'sc_reconstruction' in outputs:
            sc_loss, sc_metrics = self.sc_loss(
                outputs['features'],
                outputs['sc_reconstruction'],
                outputs['codes']
            )
            total_loss = total_loss + sc_loss
            metrics.update(sc_metrics)
        disagg_loss, disagg_metrics = self.disagg_loss(
            outputs['load_pred'], outputs['load_true'],
            outputs['pv_pred'], outputs['pv_true']
        )
        total_loss = total_loss + disagg_loss
        metrics.update(disagg_metrics)
        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics
