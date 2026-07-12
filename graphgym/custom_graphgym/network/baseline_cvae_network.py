import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

_TIME_DIM = 6


@register_network("baseline_cvae")
class BaselineCVAENetwork(nn.Module):
    """
    CVAE baseline (ported from baselines/CVAE.py's _CVAENet): a BiLSTM
    encoder produces a latent z (VAE-style), decoded into a Gaussian head
    for load, a Beta head for PV (bounded [0,1]), and a gate head
    (P(daytime)) that zero-inflates the PV distribution at night.

    Unlike earne_network.py's Encoder/Head split, this doesn't fit the
    single Head(dim_in, dim_out) -> [N, 2*n_quantiles] convention cleanly --
    the loss needs mu_z/log_var_z/gate_logit/etc, not just quantiles -- so
    it's a full @register_network implementing forward(batch) -> (pred,
    true) directly, using the register.batch stash pattern already
    established by st_sgc_caps.py / st_caps_loss.py to hand auxiliary
    tensors to cvae_loss.

    pred's first 2*n_quantiles columns are still [q_load, q_pv] (derived
    from the fitted Normal/Beta each forward call) so the shared
    earne_mae_load/earne_mae_pv metrics keep working unmodified.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        # dim_in/dim_out are unused -- create_model() always passes them,
        # but this network derives all sizing from cfg.

        self.seq_len = cfg.model.seq_len
        self.quantiles = cfg.model.quantiles

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

        enc_dim = hidden_dim * 2 + _TIME_DIM
        self.fc_mu_z = nn.Linear(enc_dim, latent_dim)
        self.fc_log_var_z = nn.Linear(enc_dim, latent_dim)

        dec_in = latent_dim + _TIME_DIM
        self.load_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim), nn.ReLU(),
            nn.Linear(decoder_dim, 2),  # mu_load, raw_sigma_load
        )
        self.solar_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim), nn.ReLU(),
            nn.Linear(decoder_dim, 2),  # raw_alpha, raw_beta
        )
        self.gate_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim // 2), nn.ReLU(),
            nn.Linear(decoder_dim // 2, 1),  # logit for P(daytime)
        )

    def _calendar(self, batch):
        # See baseline_mlp_encoder.MLPTemporalEncoder._calendar for the
        # broadcast/last-step rationale (batch.temporal is [B*T, 6]).
        N = batch.x.shape[0]
        B = batch.num_graphs
        T = self.seq_len
        temporal = batch.temporal.view(B, T, _TIME_DIM)
        nodes_per_graph = N // B
        temporal = (
            temporal.unsqueeze(1)
            .expand(B, nodes_per_graph, T, _TIME_DIM)
            .reshape(N, T, _TIME_DIM)
        )
        return temporal[:, -1, :]  # [N, 6]

    def encode(self, x_in, t):
        out, _ = self.lstm(x_in)
        h = out[:, -1, :]  # [N, hidden_dim * 2]
        ht = torch.cat([h, t], dim=-1)
        return self.fc_mu_z(ht), self.fc_log_var_z(ht)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z, t):
        zt = torch.cat([z, t], dim=-1)

        lp = self.load_head(zt)
        mu_load = lp[:, 0]
        sigma_load = F.softplus(lp[:, 1]) + 1e-4

        sp = self.solar_head(zt)
        alpha = F.softplus(sp[:, 0]) + 1e-4
        beta = F.softplus(sp[:, 1]) + 1e-4

        gate_logit = self.gate_head(zt).squeeze(-1)

        return mu_load, sigma_load, alpha, beta, gate_logit

    def _quantiles(self, mu_load, sigma_load, alpha, beta, gate_logit):
        """
        Vectorized derivation of [N, n_quantiles] load/pv quantiles from the
        fitted distributions. Runs every forward() call (train/val/test),
        unlike the original standalone CVAEBaseline.predict(), whose nested
        Python loop over scipy.stats.beta.ppf only ran once at final eval --
        scipy.stats.beta.ppf broadcasts over arrays natively, so a single
        vectorized call replaces that loop.
        """
        device = mu_load.device
        q = torch.tensor(self.quantiles, dtype=torch.float32, device=device)

        q_load = torch.distributions.Normal(
            mu_load.unsqueeze(-1), sigma_load.unsqueeze(-1)
        ).icdf(q.unsqueeze(0))  # [N, Q]

        p_day = torch.sigmoid(gate_logit)
        p_night = 1.0 - p_day

        q_np = q.cpu().numpy()[None, :]  # [1, Q]
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
        q_pv = torch.tensor(pv_q_np, dtype=torch.float32, device=device)

        return q_load, q_pv

    def forward(self, batch):
        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_in]
        if self.weather_mode:
            x_in = torch.cat([x_in, batch.weather], dim=-1)

        t = self._calendar(batch)  # [N, 6]

        mu_z, log_var_z = self.encode(x_in, t)
        z = self.reparameterize(mu_z, log_var_z) if self.training else mu_z
        mu_load, sigma_load, alpha, beta, gate_logit = self.decode(z, t)

        q_load, q_pv = self._quantiles(
            mu_load, sigma_load, alpha, beta, gate_logit
        )
        pred = torch.cat([q_load, q_pv], dim=1)  # [N, 2 * n_quantiles]

        batch.cvae_outputs = {
            "mu_load": mu_load,
            "sigma_load": sigma_load,
            "alpha": alpha,
            "beta": beta,
            "gate_logit": gate_logit,
            "mu_z": mu_z,
            "log_var_z": log_var_z,
        }
        register.batch = batch

        true = torch.stack(
            [batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1
        )
        return pred, true
