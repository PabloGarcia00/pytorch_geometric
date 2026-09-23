import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

from ._distributional_losses import masked_beta_nll, masked_bce, masked_gaussian_nll


def _kl_divergence(mu, log_var):
    return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())


@register_loss("cvae_loss")
def cvae_loss_complex(pred, true):
    """
    Loss for BaselineCVAENetwork:
      NLL_load + solar_weight * NLL_solar(daytime-masked)
               + gate_weight * BCE_gate + kl_weight * KL(mu_z, log_var_z)

    Load/solar/gate terms are only included when the corresponding target
    was actually predicted (cfg.model.predict_targets, default PV only) --
    mirrors BaselineCVAENetwork.decode() only building the heads it needs.
    KL always applies since the VAE latent is shared regardless of target
    selection.

    Unlike earne_loss.py, the tensors this needs (mu_load, sigma_load,
    alpha, beta, gate_logit, mu_z, log_var_z) don't fit through the
    (pred, true) contract, so they're read from register.batch.cvae_outputs
    -- the same stash pattern st_caps_loss.py uses for st_sgc_caps.py.

    Mask-weights loss_load/loss_gate by the dataset's NA mask (true[:, 2]),
    a deliberate parity improvement vs. the original standalone
    CVAEBaseline, which had no such mask to apply.
    """
    if cfg.model.loss_fun != "cvae_loss":
        return None

    batch = getattr(register, "batch", None)
    if batch is None or not hasattr(batch, "cvae_outputs"):
        return torch.tensor(0.0, requires_grad=True, device=pred.device), pred

    outs = batch.cvae_outputs

    y_load = true[:, 0]
    y_pv = true[:, 1]
    mask = true[:, 2]
    y_net_demand = true[:, 3]

    pv_eps = cfg.baseline.cvae_pv_eps
    daytime_threshold = cfg.baseline.cvae_daytime_threshold
    pv_clamped = y_pv.clamp(pv_eps, 1.0 - pv_eps)
    day_target = (y_pv > daytime_threshold).float()

    total_loss = _kl_divergence(outs["mu_z"], outs["log_var_z"]) * cfg.baseline.cvae_kl_weight

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
            + cfg.baseline.cvae_solar_weight * loss_solar
            + cfg.baseline.cvae_gate_weight * loss_gate
        )

        # Same PV-export physics term as earne_loss.py, added because CVAE
        # previously had no physics_weight mechanism at all. Needs its own
        # differentiable PV point-estimate -- the quantile `pred` column is
        # derived via scipy.stats.beta.ppf on detached numpy arrays
        # (BaselineCVAENetwork._quantiles), so it can't be reused here.
        # Marginal E[pv] under the zero-inflated Beta is
        # P(day) * E[Beta(alpha,beta)] = sigmoid(gate_logit) * alpha/(alpha+beta),
        # still in pv's own minmax [0,1] space -- inverse-transform (along
        # with net_demand) to physical Watts before comparing, since pv here
        # is minmax while net_demand is ihs (not directly comparable
        # otherwise, unlike the ihs/ihs pairing earne_loss.py relies on).
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
