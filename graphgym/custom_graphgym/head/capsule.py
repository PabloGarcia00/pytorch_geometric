import torch
import torch.nn as nn
from typing import Tuple

from ..layer.capsule import PrimaryCapsules, DynamicRouting


class CapsuleNetwork(nn.Module):
    def __init__(
        self,
        n_atoms: int = 85,
        n_sites: int = 40,
        capsule_dim: int = 36,
        output_dim: int = 45,
        n_iterations: int = 3,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_sites = n_sites
        self.capsule_dim = capsule_dim
        self.output_dim = output_dim
        self.n_primary_capsules = n_atoms * n_sites
        self.n_output_capsules = 2 * n_sites

        self.primary_capsules = PrimaryCapsules(n_atoms, n_sites, capsule_dim)
        self.W = nn.Parameter(torch.randn(
            self.n_output_capsules, self.n_primary_capsules, output_dim, capsule_dim
        ))
        self.routing = DynamicRouting(n_iterations)
        self.output_proj = nn.Linear(output_dim, 1)
        nn.init.xavier_normal_(self.W)

    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = codes.shape[1]
        primary = self.primary_capsules(codes)   # [n_primary, capsule_dim]
        # W: [n_out, n_primary, output_dim, capsule_dim]
        # primary: [n_primary, capsule_dim]
        # u_hat[n, p, o] = sum_d W[n, p, o, d] * primary[p, d]
        u_hat = torch.einsum('npod,pd->npo', self.W, primary)   # [n_out, n_primary, output_dim]
        output_capsules, _ = self.routing(u_hat)
        outputs = self.output_proj(output_capsules).squeeze(-1)
        return outputs[:batch_size], outputs[batch_size:]


class CapsuleRegressor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.capsule_net = CapsuleNetwork(
            n_atoms=config['q'],
            n_sites=config.get('n_sites', 40),
            capsule_dim=config['m'] + 1,
            output_dim=config['m_prime'],
            n_iterations=config['routing_iterations'],
        )

    def forward(self, codes: torch.Tensor, n_sites: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_sites is not None and n_sites != self.capsule_net.n_sites:
            device = codes.device
            self.capsule_net.n_sites = n_sites
            self.capsule_net.n_primary_capsules = self.capsule_net.n_atoms * n_sites
            self.capsule_net.n_output_capsules = 2 * n_sites
            self.capsule_net.W = nn.Parameter(torch.randn(
                self.capsule_net.n_output_capsules,
                self.capsule_net.n_primary_capsules,
                self.capsule_net.output_dim,
                self.capsule_net.capsule_dim,
                device=device,
            ))
            nn.init.xavier_normal_(self.capsule_net.W)
            self.capsule_net.primary_capsules.n_sites = n_sites
            self.capsule_net.primary_capsules.n_primary_capsules = self.capsule_net.n_atoms * n_sites
        return self.capsule_net(codes)
