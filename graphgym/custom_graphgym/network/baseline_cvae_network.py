import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets

_TIME_DIM = 6


@register_network("baseline_cvae")
class BaselineCVAENetwork(nn.Module):
    """
    CVAE baseline (ported from baselines/CVAE.py's _CVAENet): a BiLSTM
    encoder produces a latent z (VAE-style), decoded into a Gaussian head
    for load, a Beta head for PV (bounded [0,1]), and a gate head
    (P(daytime)) that zero-inflates the PV distribution at night.

    Which of load_head / (solar_head + gate_head) get built and run is
    driven by cfg.model.predict_targets (default: PV only) -- the VAE
    encoder/latent (mu_z, log_var_z) is shared regardless of target
    selection (it's one latent per household-timestep, not target-specific),
    but the decode heads are already architecturally independent sub-
    networks, so the unselected one is skipped entirely rather than merely
    excluded from the loss.

    Unlike earne_network.py's Encoder/Head split, this doesn't fit the
    single Head(dim_in, dim_out) -> [N, n_quantiles * n_targets] convention
    cleanly -- the loss needs mu_z/log_var_z/gate_logit/etc, not just
    quantiles -- so it's a full @register_network implementing
    forward(batch) -> (pred, true) directly, using the register.batch stash
    pattern already established by st_sgc_caps.py / st_caps_loss.py to hand
    auxiliary tensors to cvae_loss.

    pred's columns are still per-target quantiles (derived from the fitted
    Normal/Beta each forward call), one block per entry of active_targets(),
    so the shared earne_mae_load/earne_mae_pv metrics keep working
    unmodified.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        # dim_in/dim_out are unused -- create_model() always passes them,
        # but this network derives all sizing from cfg.

        self.seq_len = cfg.model.seq_len
        self.quantiles = cfg.model.quantiles
        self.targets = active_targets()

        self.weather_mode = cfg.earne_data.weather_mode
        self.n_weather = (
            len(cfg.earne_data.weather_features) if self.weather_mode else 0
        )
        input_size = cfg.model.dim_in + 1 + self.n_weather  # +1 operational

        latent_dim = cfg.baseline.cvae_latent_dim
        hidden_dim = cfg.baseline.cvae_hidden_dim
        decoder_dim = cfg.baseline.cvae_decoder_dim
        num_layers = cfg.baseline.cvae_num_layers
        dropout = cfg.baseline.cvae_dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Time stream: identical structure to EARNeTemporalEncoder's
        # time_encoder (stream 3) -- CNN over the full [T, 6] cyclic
        # calendar window. hidden_dim * 2 (the LSTM's output width) plays
        # the role of earne_temporal's emb_dim, so time_emb_dim follows the
        # same //4 ratio relative to it.
        time_emb_dim = (hidden_dim * 2) // 4
        self.time_encoder = nn.Sequential(
            nn.Conv1d(_TIME_DIM, 16, kernel_size=3, dilation=1, padding="same"),
            nn.GELU(),
            nn.Conv1d(16, time_emb_dim, kernel_size=3, dilation=4, padding="same"),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        enc_dim = hidden_dim * 2 + time_emb_dim
        self.fc_mu_z = nn.Linear(enc_dim, latent_dim)
        self.fc_log_var_z = nn.Linear(enc_dim, latent_dim)

        dec_in = latent_dim + time_emb_dim
        if "load" in self.targets:
            self.load_head = nn.Sequential(
                nn.Linear(dec_in, decoder_dim), nn.ReLU(),
                nn.Linear(decoder_dim, 2),  # mu_load, raw_sigma_load
            )
        if "pv" in self.targets:
            self.solar_head = nn.Sequential(
                nn.Linear(dec_in, decoder_dim), nn.ReLU(),
                nn.Linear(decoder_dim, 2),  # raw_alpha, raw_beta
            )
            self.gate_head = nn.Sequential(
                nn.Linear(dec_in, decoder_dim // 2), nn.ReLU(),
                nn.Linear(decoder_dim // 2, 1),  # logit for P(daytime)
            )

    def _time_window(self, batch):
        # batch.temporal is [B*T, 6] (one [T, 6] block per graph); broadcast
        # each graph's calendar window out to its N nodes.
        N = batch.x.shape[0]
        B = batch.num_graphs
        T = self.seq_len
        temporal = batch.temporal.view(B, T, _TIME_DIM)
        nodes_per_graph = N // B
        return (
            temporal.unsqueeze(1)
            .expand(B, nodes_per_graph, T, _TIME_DIM)
            .reshape(N, T, _TIME_DIM)
        )  # [N, T, 6]

    def encode(self, x_in, t_emb):
        out, _ = self.lstm(x_in)
        h = out[:, -1, :]  # [N, hidden_dim * 2]
        ht = torch.cat([h, t_emb], dim=-1)
        return self.fc_mu_z(ht), self.fc_log_var_z(ht)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, t_emb):
        """Returns a dict with only the entries for active_targets():
        {"mu_load", "sigma_load"} if "load" is selected, and/or
        {"alpha", "beta", "gate_logit"} if "pv" is selected.
        """
        zt = torch.cat([z, t_emb], dim=-1)
        out = {}

        if "load" in self.targets:
            lp = self.load_head(zt)
            out["mu_load"] = lp[:, 0]
            out["sigma_load"] = F.softplus(lp[:, 1]) + 1e-4

        if "pv" in self.targets:
            sp = self.solar_head(zt)
            out["alpha"] = F.softplus(sp[:, 0]) + 1e-4
            out["beta"] = F.softplus(sp[:, 1]) + 1e-4
            out["gate_logit"] = self.gate_head(zt).squeeze(-1)

        return out

    def _quantiles(self, decoded):
        """Vectorized derivation of [N, n_quantiles] quantiles per selected
        target from the fitted distributions. Runs every forward() call
        (train/val/test), unlike the original standalone
        CVAEBaseline.predict(), whose nested Python loop over
        scipy.stats.beta.ppf only ran once at final eval --
        scipy.stats.beta.ppf broadcasts over arrays natively, so a single
        vectorized call replaces that loop.

        Returns a dict with only the entries for active_targets().
        """
        device = next(iter(decoded.values())).device
        q = torch.tensor(self.quantiles, dtype=torch.float32, device=device)
        out = {}

        if "load" in self.targets:
            out["load"] = torch.distributions.Normal(
                decoded["mu_load"].unsqueeze(-1), decoded["sigma_load"].unsqueeze(-1)
            ).icdf(q.unsqueeze(0))  # [N, Q]

        if "pv" in self.targets:
            gate_logit = decoded["gate_logit"]
            p_day = torch.sigmoid(gate_logit)
            p_night = 1.0 - p_day

            q_np = q.cpu().numpy()[None, :]  # [1, Q]
            p_day_np = p_day.detach().cpu().numpy()[:, None]  # [N, 1]
            p_night_np = p_night.detach().cpu().numpy()[:, None]  # [N, 1]
            alpha_np = decoded["alpha"].detach().cpu().numpy()[:, None]  # [N, 1]
            beta_np = decoded["beta"].detach().cpu().numpy()[:, None]  # [N, 1]

            q_eff = np.clip(
                (q_np - p_night_np) / np.clip(p_day_np, 1e-6, None), 1e-6, 1 - 1e-6
            )
            pv_q_np = np.where(
                q_np > p_night_np, scipy_beta.ppf(q_eff, alpha_np, beta_np), 0.0
            )
            out["pv"] = torch.tensor(pv_q_np, dtype=torch.float32, device=device)

        return out

    def forward(self, batch):
        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_in]
        if self.weather_mode:
            x_in = torch.cat([x_in, batch.weather], dim=-1)

        temporal = self._time_window(batch)  # [N, T, 6]
        t_emb = self.time_encoder(temporal.permute(0, 2, 1))  # [N, time_emb_dim]

        mu_z, log_var_z = self.encode(x_in, t_emb)
        z = self.reparameterize(mu_z, log_var_z) if self.training else mu_z
        decoded = self.decode(z, t_emb)

        q_by_target = self._quantiles(decoded)
        pred = torch.cat([q_by_target[t] for t in self.targets], dim=1)

        batch.cvae_outputs = {**decoded, "mu_z": mu_z, "log_var_z": log_var_z}
        register.batch = batch

        true = torch.stack(
            [batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1
        )
        return pred, true
