import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

from ._distributional_losses import masked_beta_nll, masked_bce, masked_gaussian_nll


@register_loss("earne_beta_gate_loss")
def earne_beta_gate_loss(pred, true):
    """
    Loss for EARNeBetaGateHead (head/earne_beta_gate_head.py):
      NLL_load (Gaussian) + solar_weight * NLL_solar(daytime-masked, Beta)
               + gate_weight * BCE_gate

    Structurally the same per-target terms as cvae_loss.py, minus the
    KL/latent term -- EARNeBetaGateHead is a direct discriminative
    distributional regressor bolted onto earne_network.py's encoder/GNN
    embedding, not a VAE, so there's no mu_z/log_var_z to regularize.

    Reads mu_load/sigma_load/alpha/beta/gate_logit from
    register.batch.beta_gate_outputs, the stash EARNeBetaGateHead.forward()
    sets -- same stash pattern cvae_loss.py/st_caps_loss.py use for their
    own networks.
    """
    if cfg.model.loss_fun != "earne_beta_gate_loss":
        return None

    batch = getattr(register, "batch", None)
    if batch is None or not hasattr(batch, "beta_gate_outputs"):
        return torch.tensor(0.0, requires_grad=True, device=pred.device), pred

    outs = batch.beta_gate_outputs

    y_load = true[:, 0]
    y_pv = true[:, 1]
    mask = true[:, 2]
    y_net_demand = true[:, 3]

    pv_eps = cfg.beta_gate.pv_eps
    daytime_threshold = cfg.beta_gate.daytime_threshold
    pv_clamped = y_pv.clamp(pv_eps, 1.0 - pv_eps)
    day_target = (y_pv > daytime_threshold).float()

    total_loss = torch.zeros((), device=pred.device)

    if "mu_load" in outs:
        loss_load = masked_gaussian_nll(
            outs["mu_load"], outs["sigma_load"], y_load, mask
        )
        total_loss = total_loss + loss_load

    if "alpha" in outs:
        loss_solar = masked_beta_nll(
            outs["alpha"], outs["beta"], pv_clamped, day_target * mask
        )
        loss_gate = masked_bce(outs["gate_logit"], day_target, mask)
        total_loss = (
            total_loss
            + cfg.beta_gate.solar_weight * loss_solar
            + cfg.beta_gate.gate_weight * loss_gate
        )

        # Same PV-export physics term as earne_loss.py/cvae_loss.py -- see
        # cvae_loss.py's comment for the full derivation. Marginal E[pv]
        # under the zero-inflated Beta is
        # P(day) * E[Beta(alpha,beta)] = sigmoid(gate_logit) * alpha/(alpha+beta).
        physics_weight = cfg.train.physics_weight
        if physics_weight > 0.0:
            transform = register.transform
            p_day = torch.sigmoid(outs["gate_logit"])
            pv_mean_norm = p_day * (outs["alpha"] / (outs["alpha"] + outs["beta"]))
            pv_pred_watts = transform.inverse_transform("pv", pv_mean_norm)
            net_demand_watts = transform.inverse_transform("net_demand", y_net_demand)
            export_true_watts = torch.clamp(-net_demand_watts, min=0.0)
            shortfall_watts = torch.clamp(export_true_watts - pv_pred_watts, min=0.0)
            shortfall_scaled = torch.asinh(shortfall_watts)
            loss_physics = (shortfall_scaled ** 2 * mask).sum() / (mask.sum() + 1e-9)
            total_loss = total_loss + physics_weight * loss_physics

    return total_loss, pred
