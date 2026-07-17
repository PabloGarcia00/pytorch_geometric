import math

import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets
from ._baseline_common import (
    PerNodeLinearQuantileHead,
    baseline_in_features,
    build_features,
    node_ids_for_batch,
    stack_true,
)


@register_network("baseline_svr")
class BaselineSVRNetwork(nn.Module):
    """
    Per-node SVR baseline, approximated as GPU-friendly kernel quantile
    regression: a fixed Random Fourier Feature (RFF) map approximating an
    RBF kernel, followed by a per-node linear head trained with the same
    masked pinball loss as earne_network -- L2/ridge-style regularization
    comes from cfg.optim.weight_decay (already a standard GraphGym optimizer
    knob, no new config needed for it).

    Exact dual-form epsilon-insensitive SVR (e.g. sklearn.svm.SVR) is a
    CPU-only QP *and* has no native quantile/probabilistic output -- fitting
    3 independent point-estimate SVRs wouldn't produce calibrated quantiles.
    Kernel quantile regression (Takeuchi et al., 2006) generalizes the same
    kernel-machine idea to the pinball loss directly, giving native quantile
    output while staying GPU-resident and reusing the GNN's training loop
    unmodified -- see the implementation plan for the full justification.

    Bypasses earne_network's encoder/GNN-body/head chain entirely (like
    baseline_cvae_network.py / st_sgc_caps.py), so per-node fitting is
    structurally guaranteed: no message passing, no cross-household mixing.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        num_nodes = cfg.share.num_nodes
        self.weather_mode = cfg.earne_data.weather_mode
        n_weather = (
            len(cfg.earne_data.weather_features) if self.weather_mode else 0
        )
        in_features = baseline_in_features(
            cfg.model.dim_in, self.weather_mode, n_weather
        )
        # One quantile block per selected target (cfg.model.predict_targets,
        # default PV only), in active_targets() order -- see baseline_linear.py.
        out_features = len(active_targets()) * cfg.model.n_quantiles

        rff_dim = cfg.baseline.svr_rff_dim
        gamma = cfg.baseline.svr_gamma

        # Fixed (non-trainable) RFF projection approximating the RBF kernel
        # exp(-gamma * ||x - y||^2): phi(x) = sqrt(2/D) * cos(x @ W + b),
        # with W ~ N(0, 2*gamma) (Rahimi & Recht, 2007). Seeded independently
        # of the global RNG so the projection is reproducible across runs
        # regardless of cfg.seed.
        gen = torch.Generator().manual_seed(cfg.baseline.svr_seed)
        W = torch.randn(in_features, rff_dim, generator=gen) * math.sqrt(2 * gamma)
        b = torch.rand(rff_dim, generator=gen) * (2 * math.pi)
        self.register_buffer("rff_weight", W)
        self.register_buffer("rff_bias", b)
        self.rff_scale = math.sqrt(2.0 / rff_dim)

        self.num_nodes = num_nodes
        self.head = PerNodeLinearQuantileHead(num_nodes, rff_dim, out_features)

    def _rff(self, x):
        return self.rff_scale * torch.cos(x @ self.rff_weight + self.rff_bias)

    def forward(self, batch):
        x = build_features(batch, self.weather_mode, self.num_nodes)
        phi = self._rff(x)
        node_ids = node_ids_for_batch(batch, self.num_nodes)
        pred = self.head(phi, node_ids)  # [N, n_quantiles * len(targets)]
        true = stack_true(batch)
        return pred, true
