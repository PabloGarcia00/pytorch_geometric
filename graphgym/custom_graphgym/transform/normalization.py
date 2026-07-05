import numpy as np
import torch

# ── fit ───────────────────────────────────────────────────────────────


def fit_log1p(X):
    v = torch.log1p(torch.as_tensor(X, dtype=torch.float).clamp(min=0))
    return v.mean(), v.std().clamp(min=1e-8)


def fit_zscore(X):
    v = torch.as_tensor(X, dtype=torch.float)
    return v.mean(), v.std().clamp(min=1e-8)


def fit_ihs(X):
    v = torch.asinh(torch.as_tensor(X, dtype=torch.float))
    return v.mean(), v.std().clamp(min=1e-8)


def fit_minmax(X):
    v = torch.as_tensor(X, dtype=torch.float)
    lo, hi = v.min(), v.max()
    return lo, (hi - lo).clamp(min=1e-8)


# ── normalize ─────────────────────────────────────────────────────────


def norm_log1p(X, mean, std):
    v = torch.log1p(torch.as_tensor(X, dtype=torch.float).clamp(min=0))
    return (v - mean) / std


def norm_zscore(X, mean, std):
    return (torch.as_tensor(X, dtype=torch.float) - mean) / std


def norm_ihs(X, mean, std):
    v = torch.asinh(torch.as_tensor(X, dtype=torch.float))
    return (v - mean) / std


def norm_minmax(X, lo, rng, target=(0.0, 1.0)):
    v = (torch.as_tensor(X, dtype=torch.float) - lo) / rng
    return v * (target[1] - target[0]) + target[0]


# ── denormalize ───────────────────────────────────────────────────────


def denorm_log1p(Y, mean, std):
    return torch.expm1(torch.as_tensor(Y, dtype=torch.float) * std + mean)


def denorm_zscore(Y, mean, std):
    return torch.as_tensor(Y, dtype=torch.float) * std + mean


def denorm_ihs(Y, mean, std):
    return torch.sinh(torch.as_tensor(Y, dtype=torch.float) * std + mean)


def denorm_minmax(Y, lo, rng, target=(0.0, 1.0)):
    v = (torch.as_tensor(Y, dtype=torch.float) - target[0]) / (
        target[1] - target[0]
    )
    return v * rng + lo
