import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

from ..target_utils import active_targets

_quantiles_cache: dict = {}


def _get_quantiles_tensor(quantiles, device):
    key = (tuple(quantiles), str(device))
    if key not in _quantiles_cache:
        _quantiles_cache[key] = torch.tensor(
            quantiles, device=device, dtype=torch.float32
        )
    return _quantiles_cache[key]


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
        quantiles = _get_quantiles_tensor(quantiles, preds.device)

    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)

    # Average over quantiles, then apply mask
    # loss.mean(dim=-1) results in [N]
    masked_loss = loss.mean(dim=-1) * mask

    # Return average over non-masked elements
    return masked_loss.sum() / (mask.sum() + 1e-9)


@register_loss("earne_loss")
def earne_loss_complex(pred, true):
    """
    Custom loss for EARNe:
    1. Quantile loss for each target selected via cfg.model.predict_targets
       (Load and/or PV -- default PV only)
    2. Physics-informed constraint: predicted PV must be at least the export
       implied by net demand (you can't export more than you generate) --
       needs only PV, matching the PV-only default (see PhysicsLoss-era
       load-pv=net_demand constraint this replaced, which needed both targets)
    3. Crossing penalty, applied per selected target independently
    """
    if cfg.model.loss_fun != "earne_loss":
        return None

    quantiles = cfg.model.quantiles
    physics_weight = cfg.train.physics_weight
    crossing_weight = cfg.train.crossing_weight
    n_q = len(quantiles)
    targets = active_targets()

    true_by_target = {"load": true[:, 0], "pv": true[:, 1]}
    mask = true[:, 2]
    y_net_demand = true[:, 3]

    q_by_target = {
        t: pred[:, i * n_q : (i + 1) * n_q] for i, t in enumerate(targets)
    }

    total_loss = 0.0
    for t, q in q_by_target.items():
        total_loss = total_loss + masked_quantile_loss(
            q, true_by_target[t], quantiles, mask
        )

    assert 0.5 in quantiles, "median must be in quantiles!"
    q50_idx = quantiles.index(0.5)

    # Physics loss: predicted PV must be >= the export implied by net demand
    # (export = max(-net_demand, 0) -- generation net of consumption; you
    # cannot export more than you generate). Mirrors the eval-time
    # export_violation metric (regression.py) exactly, so this loss
    # directly targets what that metric measures. Only needs PV, so it's
    # well-defined under the PV-only default.
    #
    # q_by_target/y_net_demand are each independently normalized (their own
    # energy_norm_mode), so subtracting them directly isn't physically
    # meaningful -- inverse-transform both to physical Watts first. The
    # resulting Watt-scale shortfall is then asinh-compressed (same
    # log-like compression every other energy quantity in this codebase
    # uses) before squaring, so physics_weight stays in a comparable range
    # to the ihs-scale quantile loss rather than being dominated by raw
    # Watt^2 magnitudes.
    if physics_weight > 0.0 and "pv" in q_by_target:
        transform = register.transform
        pv_pred_watts = transform.inverse_transform(
            "pv", q_by_target["pv"][:, q50_idx]
        )
        net_demand_watts = transform.inverse_transform("net_demand", y_net_demand)
        export_true_watts = torch.clamp(-net_demand_watts, min=0.0)
        shortfall_watts = torch.clamp(export_true_watts - pv_pred_watts, min=0.0)
        shortfall_scaled = torch.asinh(shortfall_watts)
        loss_physics = (shortfall_scaled ** 2 * mask).sum() / (mask.sum() + 1e-9)
        total_loss = total_loss + physics_weight * loss_physics

    # Crossing penalty — penalise q[i] > q[i+1], per selected target
    if crossing_weight > 0.0:
        loss_crossing = 0.0
        for q in q_by_target.values():
            for i in range(n_q - 1):
                loss_crossing += torch.clamp(
                    q[:, i] - q[:, i + 1], min=0.0
                ).mean()
        total_loss = total_loss + crossing_weight * loss_crossing

    return total_loss, pred
