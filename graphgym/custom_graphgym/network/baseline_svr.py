import math

import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets
from ._baseline_common import (
    PerNodeLinearQuantileHead,
    baseline_current_in_features,
    baseline_in_features,
    build_current_features,
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

    cfg.baseline.svr_feature_mode selects the input the RFF kernel sees:
    'window' (default) flattens a [seq_len, C] history window (+ calendar
    anchor); 'current' uses only the most recent timestep's raw signal
    (+ calendar anchor) -- see results_archives/ for the write-up.

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
        self.feature_mode = cfg.baseline.svr_feature_mode
        assert self.feature_mode in ("window", "current"), (
            f"Unknown svr_feature_mode: {self.feature_mode!r}"
        )
        if self.feature_mode == "window":
            in_features = baseline_in_features(
                cfg.model.dim_in, self.weather_mode, n_weather
            )
        else:
            in_features = baseline_current_in_features(
                cfg.model.dim_in, self.weather_mode, n_weather
            )
        # One quantile block per selected target (cfg.model.predict_targets,
        # default PV only), in active_targets() order -- see baseline_linear.py.
        out_features = len(active_targets()) * cfg.model.n_quantiles

        self.rff_dim = cfg.baseline.svr_rff_dim
        self.in_features = in_features
        self.gamma_auto = cfg.baseline.svr_gamma_auto
        self.svr_seed = cfg.baseline.svr_seed
        self.rff_scale = math.sqrt(2.0 / self.rff_dim)
        self._rff_fitted = False

        # Buffer slots always exist from __init__, even under gamma_auto
        # (placeholder zeros, overwritten by the real fit on first forward)
        # -- otherwise checkpoint loading breaks: create_model() builds a
        # fresh instance and calls load_state_dict() *without* a forward
        # pass first (see notebooks/calculate_metrics.py's
        # disaggregate_test_set), so a checkpoint saved after lazy-fitting
        # has "rff_weight"/"rff_bias" keys that a never-forwarded fresh
        # instance has no matching buffer for -> "Unexpected key(s)".
        self.register_buffer("rff_weight", torch.zeros(in_features, self.rff_dim))
        self.register_buffer("rff_bias", torch.zeros(self.rff_dim))
        # A loaded checkpoint's rff_weight/bias are already the real fitted
        # values -- mark fitted so the next forward() doesn't clobber them
        # with a fresh (and for eval-only batches, wrongly-scaled) auto-fit.
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: setattr(module, "_rff_fitted", True)
        )

        if not self.gamma_auto:
            # Fixed gamma, same as before: RFF projection approximating the
            # RBF kernel exp(-gamma * ||x - y||^2), phi(x) = sqrt(2/D) *
            # cos(x @ W + b), W ~ N(0, 2*gamma) (Rahimi & Recht, 2007).
            self._build_rff(cfg.baseline.svr_gamma)

        self.num_nodes = num_nodes
        self.head = PerNodeLinearQuantileHead(num_nodes, self.rff_dim, out_features)

    def _build_rff(self, gamma: float, device=None) -> None:
        gen = torch.Generator().manual_seed(self.svr_seed)
        W = torch.randn(self.in_features, self.rff_dim, generator=gen) * math.sqrt(2 * gamma)
        b = torch.rand(self.rff_dim, generator=gen) * (2 * math.pi)
        if device is not None:
            W, b = W.to(device), b.to(device)
        with torch.no_grad():
            self.rff_weight.copy_(W)
            self.rff_bias.copy_(b)
        self._rff_fitted = True

    def _rff(self, x):
        if not self._rff_fitted:
            # gamma='scale' (sklearn's default): 1 / (n_features * X.var()).
            # RFF needs gamma to build W before any data pass, so this fits
            # on the first batch actually seen (~32 graphs x num_nodes rows,
            # large enough for a stable variance estimate) rather than
            # needing a separate pre-pass like baseline_knn.py's fit_cache.
            var = x.detach().var().clamp(min=1e-8)
            gamma = 1.0 / (self.in_features * var.item())
            self._build_rff(gamma, device=x.device)
        return self.rff_scale * torch.cos(x @ self.rff_weight + self.rff_bias)

    def forward(self, batch):
        if self.feature_mode == "window":
            x = build_features(batch, self.weather_mode, self.num_nodes)
        else:
            x = build_current_features(batch, self.weather_mode, self.num_nodes)
        phi = self._rff(x)
        node_ids = node_ids_for_batch(batch, self.num_nodes)
        pred = self.head(phi, node_ids)  # [N, n_quantiles * len(targets)]
        true = stack_true(batch)
        return pred, true
