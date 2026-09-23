"""Shared Gaussian (load) + zero-inflated-Beta (PV, gated by P(daytime))
predictive-distribution math.

Originally implemented once, inline, inside BaselineCVAENetwork (network/
baseline_cvae_network.py) -- decode_gaussian/decode_beta_params replace
that class's decode()'s per-target arithmetic, and gaussian_quantiles/
zero_inflated_beta_quantiles replace its _quantiles()'s. Factored out here
so EARNeBetaGateHead (head/earne_beta_gate_head.py) -- which ports the same
technique onto earne_network.py's ST-GNN/LSTM/MLP shell via a plain
cfg.model.head_name swap -- shares one implementation with the CVAE instead
of a second copy that could silently drift from it.

Every function is elementwise/last-dim-only, so callers may pass tensors
with extra leading batch dims (e.g. BaselineCVAENetwork._mc_quantiles's
[K, N, ...] Monte-Carlo samples) without any change here.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta

_SOFTPLUS_EPS = 1e-4


def decode_gaussian(raw):
    """raw: [..., 2] (mu, raw_sigma) -> (mu, sigma), sigma > 0 via softplus."""
    mu = raw[..., 0]
    sigma = F.softplus(raw[..., 1]) + _SOFTPLUS_EPS
    return mu, sigma


def decode_beta_params(raw):
    """raw: [..., 2] (raw_alpha, raw_beta) -> (alpha, beta), both > 0."""
    alpha = F.softplus(raw[..., 0]) + _SOFTPLUS_EPS
    beta = F.softplus(raw[..., 1]) + _SOFTPLUS_EPS
    return alpha, beta


def gaussian_quantiles(mu, sigma, q):
    """mu/sigma: [...] (e.g. [N] or [K, N]), q: [Q] -> [..., Q]."""
    return torch.distributions.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1)).icdf(
        q.view(*([1] * mu.dim()), -1)
    )


def zero_inflated_beta_quantiles(alpha, beta, gate_logit, q):
    """Analytic quantiles of the zero-inflated Beta: P(day) = sigmoid(gate_logit)
    mass on Beta(alpha, beta), P(night) = 1 - P(day) mass at exactly 0.

    alpha/beta/gate_logit: [N] (only -- unlike the other helpers here, the
    scipy.stats.beta.ppf call this needs doesn't support extra leading
    batch dims; BaselineCVAENetwork._mc_quantiles uses its own
    torch.quantile-based empirical path for the [K, N] case instead of this
    function). q: [Q]. Returns [N, Q].
    """
    device = alpha.device
    q_np = q.detach().cpu().numpy()[None, :]  # [1, Q]

    p_day = torch.sigmoid(gate_logit)
    p_night = 1.0 - p_day
    p_day_np = p_day.detach().cpu().numpy()[:, None]  # [N, 1]
    p_night_np = p_night.detach().cpu().numpy()[:, None]  # [N, 1]
    alpha_np = alpha.detach().cpu().numpy()[:, None]  # [N, 1]
    beta_np = beta.detach().cpu().numpy()[:, None]  # [N, 1]

    q_eff = np.clip(
        (q_np - p_night_np) / np.clip(p_day_np, 1e-6, None), 1e-6, 1 - 1e-6
    )
    pv_q_np = np.where(
        q_np > p_night_np, scipy_beta.ppf(q_eff, alpha_np, beta_np), 0.0
    )
    return torch.tensor(pv_q_np, dtype=torch.float32, device=device)
