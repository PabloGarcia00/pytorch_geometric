"""Shared masked-NLL/BCE terms for distributional (Gaussian/Beta/gate)
heads. Originally private to cvae_loss.py; factored out so
earne_beta_gate_loss.py (which trains EARNeBetaGateHead -- the same
Gaussian-load + zero-inflated-Beta-PV technique ported onto earne_network.py
via cfg.model.head_name) shares one implementation instead of a second copy
that could silently drift from it.

Leading underscore: not meant to be selected via cfg.model.loss_fun itself
(it registers nothing), just imported by the loss functions that are.
"""
import torch
import torch.nn.functional as F


def masked_gaussian_nll(mu, sigma, target, mask):
    denom = mask.sum()
    if denom < 1:
        return torch.zeros((), device=mu.device)
    nll = -torch.distributions.Normal(mu, sigma).log_prob(target)
    return (nll * mask).sum() / denom


def masked_beta_nll(alpha, beta, target, mask):
    denom = mask.sum()
    if denom < 1:
        return torch.zeros((), device=alpha.device)
    nll = -torch.distributions.Beta(alpha, beta).log_prob(target)
    return (nll * mask).sum() / denom


def masked_bce(logit, target, mask):
    denom = mask.sum()
    if denom < 1:
        return torch.zeros((), device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (bce * mask).sum() / denom
