import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg

def masked_quantile_loss(preds, targets, quantiles, mask):
    """
    Implementation of the Pinball Loss (Quantile Loss)
    preds: [N, n_quantiles]
    targets: [N]
    quantiles: [n_quantiles]
    mask: [N] (1.0 for valid data, 0.0 for NA)
    """
    # Ensure targets is [N, 1] for broadcasting
    targets = targets.unsqueeze(-1)
    
    # Calculate error: [N, n_quantiles]
    errors = targets - preds
    
    # Pinball loss: L = max(q * e, (q - 1) * e)
    if not isinstance(quantiles, torch.Tensor):
        quantiles = torch.tensor(quantiles, device=preds.device, dtype=torch.float32)
        
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    
    # Average over quantiles, then apply mask
    # loss.mean(dim=-1) results in [N]
    masked_loss = (loss.mean(dim=-1) * mask)
    
    # Return average over non-masked elements
    return masked_loss.sum() / (mask.sum() + 1e-9)

@register_loss('earne_loss')
def earne_loss_complex(pred, true):
    if cfg.model.loss_fun != 'earne_loss':
        return None

    """
    Custom loss for EARNe:
    1. Quantile loss for Load
    2. Quantile loss for PV
    3. Physics-informed constraint: Load - PV = NetDemand
    """
    # GraphGym typically passes (pred, true), but for complex tasks
    # we often need to pull intermediate tensors from the batch object
    # assuming the model/head attached them there.

    # Note: 'pred' and 'true' in GraphGym are usually Tensors,
    # but some custom setups might use the batch object.
    # We'll pull from cfg or the global batch if available, 
    # but here we assume the standard pred/true are passed or accessible.

    # true: [N, 4] columns -> (y_load, y_pv, mask, net_demand)
    # pred: [N, 2 * n_quantiles]
    # q_load / q_pv are also attached to the batch by the head for convenience.
    import torch_geometric.graphgym.register as register
    batch = register.batch if hasattr(register, 'batch') else None

    if batch is None:
        return torch.tensor(0.0, requires_grad=True, device=pred.device), pred

    quantiles = cfg.model.quantiles
    physics_weight = cfg.train.physics_weight

    y_load = true[:, 0]
    y_pv = true[:, 1]
    mask = true[:, 2]
    y_net_demand = true[:, 3]

    # 1. Load Loss
    loss_load = masked_quantile_loss(batch.q_load, y_load, quantiles, mask)

    # 2. PV Loss
    loss_pv = masked_quantile_loss(batch.q_pv, y_pv, quantiles, mask)

    # 3. Physics Loss (median quantile only)
    try:
        q50_idx = quantiles.index(0.5)
        pred_net_demand = batch.q_load[:, q50_idx] - batch.q_pv[:, q50_idx]
        
        # Calculate MSE only on valid (non-masked) entries
        diff_sq = (pred_net_demand - y_net_demand)**2
        loss_physics = (diff_sq * mask).sum() / (mask.sum() + 1e-9)
    except (ValueError, IndexError):
        loss_physics = 0.0

    total_loss = loss_load + loss_pv + (physics_weight * loss_physics)
    
    return total_loss, pred
