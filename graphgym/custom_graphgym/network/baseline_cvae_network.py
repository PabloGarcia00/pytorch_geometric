import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..distributions.beta_gate import (
    decode_beta_params,
    decode_gaussian,
    gaussian_quantiles,
    zero_inflated_beta_quantiles,
)
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

    Uncertainty quantification: by default (cfg.baseline.cvae_mc_samples <=
    1) eval decodes a single deterministic z=mu_z and derives quantiles
    analytically (icdf/ppf) -- aleatoric-only, from the decoder heads.
    Setting cvae_mc_samples > 1 switches eval to Monte Carlo predictive
    quantiles (_mc_quantiles): multiple z_k ~ N(mu_z, sigma_z) are sampled
    from the VAE posterior (epistemic), each decoded and sampled from
    (aleatoric), and the pooled draws' empirical quantiles become pred --
    same shape/columns either way, so nothing downstream needs to know
    which path ran. Training is unaffected either way (always a single
    reparameterized z, per the original VAE training objective).
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

        z/t_emb may carry an extra leading MC-sample dim ([K, N, ...], from
        _mc_quantiles) on top of the usual [N, ...] -- every op here is
        elementwise or acts on the last dim, so indexing uses `...` rather
        than `:` to work unchanged in both cases.
        """
        zt = torch.cat([z, t_emb], dim=-1)
        out = {}

        if "load" in self.targets:
            out["mu_load"], out["sigma_load"] = decode_gaussian(self.load_head(zt))

        if "pv" in self.targets:
            out["alpha"], out["beta"] = decode_beta_params(self.solar_head(zt))
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
            out["load"] = gaussian_quantiles(
                decoded["mu_load"], decoded["sigma_load"], q
            )  # [N, Q]

        if "pv" in self.targets:
            out["pv"] = zero_inflated_beta_quantiles(
                decoded["alpha"], decoded["beta"], decoded["gate_logit"], q
            )  # [N, Q]

        return out

    def _predictive_samples(self, decoded):
        """One Monte Carlo draw per active target from the fitted
        per-sample distributions in `decoded` (any leading batch shape,
        e.g. [K, N]). Zero-inflates the PV draw by its own sampled daytime
        indicator so MC samples respect the same day/night structure as
        _quantiles()'s analytic zero-inflated Beta.
        """
        out = {}

        if "load" in self.targets:
            out["load"] = torch.distributions.Normal(
                decoded["mu_load"], decoded["sigma_load"]
            ).sample()

        if "pv" in self.targets:
            is_day = torch.bernoulli(torch.sigmoid(decoded["gate_logit"]))
            pv_draw = torch.distributions.Beta(
                decoded["alpha"], decoded["beta"]
            ).sample()
            out["pv"] = is_day * pv_draw

        return out

    def _mc_quantiles(self, mu_z, log_var_z, t_emb):
        """Monte Carlo predictive quantiles: draws cfg.baseline.cvae_mc_samples
        latent codes z_k ~ N(mu_z, sigma_z) -- the epistemic spread of the
        CVAE's own posterior, which is otherwise collapsed to the point
        estimate mu_z everywhere else in this module at eval time -- decodes
        each into its own Normal/zero-inflated-Beta head, then draws one
        predictive sample per target from each decode (aleatoric spread).
        The pooled K samples therefore carry both uncertainty sources;
        empirical quantiles (torch.quantile) replace the closed-form
        icdf/ppf of _quantiles(), which only ever sees a single mu_z decode.

        Returns a dict with only the entries for active_targets(), each
        [N, Q] -- same shape/order _quantiles() returns, so forward()'s
        pred assembly doesn't care which path produced it.
        """
        K = cfg.baseline.cvae_mc_samples
        std_z = torch.exp(0.5 * log_var_z)
        eps = torch.randn(K, *mu_z.shape, device=mu_z.device, dtype=mu_z.dtype)
        z_samples = mu_z.unsqueeze(0) + std_z.unsqueeze(0) * eps  # [K, N, latent_dim]
        t_emb_k = t_emb.unsqueeze(0).expand(K, *t_emb.shape)  # [K, N, time_emb_dim]

        decoded_k = self.decode(z_samples, t_emb_k)  # each value: [K, N]
        samples = self._predictive_samples(decoded_k)  # each value: [K, N]

        q = torch.tensor(self.quantiles, dtype=torch.float32, device=mu_z.device)
        return {
            name: torch.quantile(y, q, dim=0).transpose(0, 1)  # [N, Q]
            for name, y in samples.items()
        }

    def forward(self, batch):
        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_in]
        if self.weather_mode:
            x_in = torch.cat([x_in, batch.weather], dim=-1)

        temporal = self._time_window(batch)  # [N, T, 6]
        t_emb = self.time_encoder(temporal.permute(0, 2, 1))  # [N, time_emb_dim]

        mu_z, log_var_z = self.encode(x_in, t_emb)
        z = self.reparameterize(mu_z, log_var_z) if self.training else mu_z
        decoded = self.decode(z, t_emb)

        if not self.training and cfg.baseline.cvae_mc_samples > 1:
            q_by_target = self._mc_quantiles(mu_z, log_var_z, t_emb)
        else:
            q_by_target = self._quantiles(decoded)
        pred = torch.cat([q_by_target[t] for t in self.targets], dim=1)

        batch.cvae_outputs = {**decoded, "mu_z": mu_z, "log_var_z": log_var_z}
        register.batch = batch

        true = torch.stack(
            [batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1
        )
        return pred, true
