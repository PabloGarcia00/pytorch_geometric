import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head

from ..distributions.beta_gate import (
    decode_beta_params,
    decode_gaussian,
    gaussian_quantiles,
    zero_inflated_beta_quantiles,
)
from ..target_utils import active_targets


@register_head("earne_beta_gate")
class EARNeBetaGateHead(nn.Module):
    r"""
    Drop-in alternative to EARNeQuantileHead (earne_head.py): instead of
    directly regressing each quantile (a plain nn.Linear(dim_in,
    n_quantiles) per target, trained with pinball loss), this fits a
    parametric predictive distribution per target from the node embedding
    -- Normal for load, zero-inflated Beta for PV (bounded [0,1], with a
    gate_logit modeling P(daytime) so night-time PV collapses to exactly
    0) -- and derives quantiles analytically. Ported from
    BaselineCVAENetwork.decode()/_quantiles() (network/
    baseline_cvae_network.py), which established this technique for the
    CVAE baseline; this head makes it available to any earne_network.py
    variant (ST-GNN, LSTM/MLP baselines) purely via a cfg.model.head_name
    swap -- no changes to earne_network.py itself.

    Must be paired with loss_fun: earne_beta_gate_loss (loss/
    earne_beta_gate_loss.py) -- earne_loss's pinball loss has nothing to
    train mu_load/sigma_load/alpha/beta/gate_logit with.

    Like EARNeQuantileHead, dim_out is unused (create_model() always
    passes it, but sizing here comes entirely from dim_in -- the GNN/
    encoder's node embedding width -- plus cfg.model.n_quantiles/quantiles
    and active_targets()). Unlike BaselineCVAENetwork's decoder heads
    (which decode from a small VAE latent + a small time embedding, so
    need a hidden layer), this decodes straight off dim_in the same way
    EARNeQuantileHead does, so each target gets a single nn.Linear, no
    hidden layer.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.quantiles = cfg.model.quantiles
        self.targets = active_targets()

        if "load" in self.targets:
            self.load_head = nn.Linear(dim_in, 2)  # mu_load, raw_sigma_load
        if "pv" in self.targets:
            self.solar_head = nn.Linear(dim_in, 2)  # raw_alpha, raw_beta
            self.gate_head = nn.Linear(dim_in, 1)  # logit for P(daytime)

    def _decode(self, x):
        """Returns a dict with only the entries for active_targets():
        {"mu_load", "sigma_load"} if "load" is selected, and/or
        {"alpha", "beta", "gate_logit"} if "pv" is selected.
        """
        out = {}

        if "load" in self.targets:
            out["mu_load"], out["sigma_load"] = decode_gaussian(self.load_head(x))

        if "pv" in self.targets:
            out["alpha"], out["beta"] = decode_beta_params(self.solar_head(x))
            out["gate_logit"] = self.gate_head(x).squeeze(-1)

        return out

    def _quantiles(self, decoded):
        """Vectorized [N, n_quantiles] quantiles per selected target from
        the fitted distributions -- identical math to
        BaselineCVAENetwork._quantiles(), via the shared distributions/
        beta_gate.py helpers.
        """
        device = next(iter(decoded.values())).device
        q = torch.tensor(self.quantiles, dtype=torch.float32, device=device)
        out = {}

        if "load" in self.targets:
            out["load"] = gaussian_quantiles(
                decoded["mu_load"], decoded["sigma_load"], q
            )  # [N, Q]

        if "pv" in self.targets:
            out["pv"] = zero_inflated_beta_quantiles(
                decoded["alpha"], decoded["beta"], decoded["gate_logit"], q
            )  # [N, Q]

        return out

    def forward(self, batch):
        # batch.x contains the embeddings from the encoder/GNN body.
        decoded = self._decode(batch.x)
        q_by_target = self._quantiles(decoded)

        # Stash for earne_beta_gate_loss, same register.batch pattern
        # BaselineCVAENetwork/st_sgc_caps.py use to hand auxiliary tensors
        # to their losses -- earne_network.py passes `batch` into this head
        # by reference and never sets register.batch itself, so it's safe
        # to set here without touching earne_network.py.
        batch.beta_gate_outputs = decoded
        register.batch = batch

        # Same [N, n_quantiles * len(targets)] convention EARNeQuantileHead
        # returns, load-before-pv when both selected (active_targets()) --
        # so earne_mae_load/pv and DisaggregationMetrics need no changes.
        return torch.cat([q_by_target[t] for t in self.targets], dim=1)
