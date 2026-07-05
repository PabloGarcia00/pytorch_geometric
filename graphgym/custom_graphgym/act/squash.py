import torch


def squash(s: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(s, dim=-1, keepdim=True)
    norm_sq = norm ** 2
    return (norm_sq / (1 + norm_sq)) * (s / (norm + 1e-8))
