import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..act.squash import squash


class PrimaryCapsules(nn.Module):
    def __init__(self, n_atoms: int = 85, n_sites: int = 40, capsule_dim: int = 36):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_sites = n_sites
        self.capsule_dim = capsule_dim
        self.n_primary_capsules = n_atoms * n_sites

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        T, N, q = codes.shape
        assert T == self.capsule_dim, f"Expected T={self.capsule_dim}, got T={T}"
        codes_transposed = codes.transpose(1, 2)        # [T, q, N]
        codes_reshaped = codes_transposed.reshape(T, q * N)  # [T, N*q]
        return codes_reshaped.transpose(0, 1)           # [(N*q), T]


class DynamicRouting(nn.Module):
    def __init__(self, n_iterations: int = 3):
        super().__init__()
        self.n_iterations = n_iterations

    def forward(self, u_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_out, n_primary, dim = u_hat.shape
        b = torch.zeros(n_out, n_primary, device=u_hat.device)

        for iteration in range(self.n_iterations):
            c = F.softmax(b, dim=0)
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)
            v = squash(s)
            if iteration < self.n_iterations - 1:
                delta_b = torch.sum(u_hat * v.unsqueeze(1), dim=-1)
                b = b + delta_b

        return v, c
