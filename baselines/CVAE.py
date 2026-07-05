"""CVAE baseline: shared BiLSTM encoder with dual probabilistic heads.

Architecture:
  Input: net load sequence + cyclic time embeddings
  Shared Encoder: BiLSTM → reparameterized latent Z (VAE-style)
  Head 1 (Load):  Gaussian NLL  → predicts μ, σ
  Head 2 (Solar): Beta NLL      → predicts α, β  (bounded [0,1], U-shape)
  Gate Head:      BCE            → predicts P(daytime) to zero-out solar at night

Total loss: NLL_load + w_solar * NLL_solar + w_gate * BCE_gate + w_kl * KL
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta
from torch.utils.data import DataLoader, TensorDataset

TIME_DIM = 6  # sin/cos encodings for hour-of-day, day-of-week, day-of-year


# ── Time features ────────────────────────────────────────────────────────────

def _cyclic_time_features(timestamps) -> torch.Tensor:
    """Convert timestamps to [S, 6] cyclic features."""
    dt = pd.to_datetime(timestamps)
    h   = dt.hour.values.astype(np.float32)
    dow = dt.dayofweek.values.astype(np.float32)
    doy = dt.dayofyear.values.astype(np.float32)
    feats = np.stack([
        np.sin(2 * np.pi * h   / 24),
        np.cos(2 * np.pi * h   / 24),
        np.sin(2 * np.pi * dow /  7),
        np.cos(2 * np.pi * dow /  7),
        np.sin(2 * np.pi * doy / 365),
        np.cos(2 * np.pi * doy / 365),
    ], axis=-1)
    return torch.tensor(feats, dtype=torch.float32)


# ── Loss functions ────────────────────────────────────────────────────────────

def _gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -torch.distributions.Normal(mu, sigma).log_prob(target).mean()


def _beta_nll(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Beta NLL evaluated only on daytime samples (mask==1)."""
    denom = mask.sum()
    if denom < 1:
        return torch.zeros(1, device=alpha.device).squeeze()
    nll = -torch.distributions.Beta(alpha, beta).log_prob(target)
    return (nll * mask).sum() / denom


def _kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())


# ── Neural network ────────────────────────────────────────────────────────────

class _CVAENet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        decoder_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        enc_dim = hidden_dim * 2 + TIME_DIM
        self.fc_mu_z     = nn.Linear(enc_dim, latent_dim)
        self.fc_log_var_z = nn.Linear(enc_dim, latent_dim)

        dec_in = latent_dim + TIME_DIM

        self.load_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim), nn.ReLU(),
            nn.Linear(decoder_dim, 2),   # mu_load, raw_sigma_load
        )
        self.solar_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim), nn.ReLU(),
            nn.Linear(decoder_dim, 2),   # raw_alpha, raw_beta
        )
        self.gate_head = nn.Sequential(
            nn.Linear(dec_in, decoder_dim // 2), nn.ReLU(),
            nn.Linear(decoder_dim // 2, 1),      # logit for P(daytime)
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor):
        out, _ = self.lstm(x.unsqueeze(-1))
        h = out[:, -1, :]                        # [B, hidden*2]
        ht = torch.cat([h, t], dim=-1)
        return self.fc_mu_z(ht), self.fc_log_var_z(ht)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, t: torch.Tensor):
        zt = torch.cat([z, t], dim=-1)

        lp = self.load_head(zt)
        mu_load    = lp[:, 0]
        sigma_load = F.softplus(lp[:, 1]) + 1e-4

        sp = self.solar_head(zt)
        alpha = F.softplus(sp[:, 0]) + 1e-4
        beta  = F.softplus(sp[:, 1]) + 1e-4

        gate_logit = self.gate_head(zt).squeeze(-1)

        return mu_load, sigma_load, alpha, beta, gate_logit

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        mu_z, log_var_z = self.encode(x, t)
        z = self.reparameterize(mu_z, log_var_z)
        mu_load, sigma_load, alpha, beta, gate_logit = self.decode(z, t)
        return mu_load, sigma_load, alpha, beta, gate_logit, mu_z, log_var_z


# ── Baseline wrapper ──────────────────────────────────────────────────────────

class CVAEBaseline:
    """
    CVAE with Gaussian load head, Beta solar head, and daytime gate.

    Inputs:
        x_train:          [S, N, seq_len]  net load history (scaled)
        y_load_train:     [S, N]           load targets (scaled)
        y_pv_train:       [S, N]           pv targets (scaled, ≥ 0)
        timestamps_train: [S]              pandas/numpy timestamps

    Outputs (predict):
        {"load": [S, N, Q], "pv": [S, N, Q]}   quantile predictions
    """

    def __init__(
        self,
        seq_len: int = 96,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        decoder_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = None,
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
        solar_weight: float = 1.0,
        gate_weight: float = 1.0,
        kl_weight: float = 0.1,
        daytime_threshold: float = 0.005,
        pv_eps: float = 0.001,
        device: str = "auto",
    ):
        self.seq_len           = seq_len
        self.latent_dim        = latent_dim
        self.hidden_dim        = hidden_dim
        self.decoder_dim       = decoder_dim
        self.num_layers        = num_layers
        self.dropout           = dropout
        self.quantiles         = quantiles or [0.1, 0.5, 0.9]
        self.n_epochs          = n_epochs
        self.lr                = lr
        self.batch_size        = batch_size
        self.solar_weight      = solar_weight
        self.gate_weight       = gate_weight
        self.kl_weight         = kl_weight
        self.daytime_threshold = daytime_threshold   # fraction of pv_max
        self.pv_eps            = pv_eps              # clip margin for Beta
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self._model: _CVAENet | None = None
        self.pv_max: float = 1.0

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        x_train: torch.Tensor,
        y_load_train: torch.Tensor,
        y_pv_train: torch.Tensor,
        timestamps_train,
        x_val: Optional[torch.Tensor] = None,
        y_load_val: Optional[torch.Tensor] = None,
        y_pv_val: Optional[torch.Tensor] = None,
        timestamps_val=None,
    ) -> "CVAEBaseline":
        S, N, L = x_train.shape

        self.pv_max = float(y_pv_train.max().clamp(min=1e-6))

        t_train = _cyclic_time_features(timestamps_train)          # [S, TIME_DIM]
        t_flat  = t_train.unsqueeze(1).expand(S, N, TIME_DIM).reshape(S * N, TIME_DIM)
        x_flat  = x_train.reshape(S * N, L)
        load_flat = y_load_train.reshape(S * N)

        pv_norm   = (y_pv_train / self.pv_max).clamp(self.pv_eps, 1.0 - self.pv_eps)
        pv_flat   = pv_norm.reshape(S * N)
        day_flat  = (y_pv_train / self.pv_max > self.daytime_threshold).float().reshape(S * N)

        self._model = _CVAENet(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            decoder_dim=self.decoder_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        dataset   = TensorDataset(x_flat, t_flat, load_flat, pv_flat, day_flat)
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.n_epochs + 1):
            self._model.train()
            total_loss = 0.0

            for xb, tb, lb, pb, db in loader:
                xb, tb, lb, pb, db = (v.to(self.device) for v in (xb, tb, lb, pb, db))

                mu_load, sigma_load, alpha, beta, gate_logit, mu_z, log_var_z = self._model(xb, tb)

                loss_load  = _gaussian_nll(mu_load, sigma_load, lb)
                loss_solar = _beta_nll(alpha, beta, pb, db)
                loss_gate  = F.binary_cross_entropy_with_logits(gate_logit, db)
                loss_kl    = _kl_divergence(mu_z, log_var_z)

                loss = (
                    loss_load
                    + self.solar_weight * loss_solar
                    + self.gate_weight  * loss_gate
                    + self.kl_weight    * loss_kl
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / (S * N)
            val_str  = ""
            if x_val is not None and timestamps_val is not None:
                val_loss = self._eval_loss(x_val, y_load_val, y_pv_val, timestamps_val)
                val_str  = f"  val_loss={val_loss:.4f}"

            print(f"epoch {epoch:3d}/{self.n_epochs}  train_loss={avg_loss:.4f}{val_str}")

        return self

    def _eval_loss(self, x_val, y_load_val, y_pv_val, timestamps_val) -> float:
        S, N, L = x_val.shape
        t_val   = _cyclic_time_features(timestamps_val)
        t_flat  = t_val.unsqueeze(1).expand(S, N, TIME_DIM).reshape(S * N, TIME_DIM)
        x_flat  = x_val.reshape(S * N, L)
        load_flat = y_load_val.reshape(S * N)
        pv_norm   = (y_pv_val / self.pv_max).clamp(self.pv_eps, 1.0 - self.pv_eps)
        pv_flat   = pv_norm.reshape(S * N)
        day_flat  = (y_pv_val / self.pv_max > self.daytime_threshold).float().reshape(S * N)

        self._model.eval()
        total_loss = 0.0
        dataset = TensorDataset(x_flat, t_flat, load_flat, pv_flat, day_flat)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for xb, tb, lb, pb, db in loader:
                xb, tb, lb, pb, db = (v.to(self.device) for v in (xb, tb, lb, pb, db))
                mu_load, sigma_load, alpha, beta, gate_logit, mu_z, log_var_z = self._model(xb, tb)
                loss = (
                    _gaussian_nll(mu_load, sigma_load, lb)
                    + self.solar_weight * _beta_nll(alpha, beta, pb, db)
                    + self.gate_weight  * F.binary_cross_entropy_with_logits(gate_logit, db)
                    + self.kl_weight    * _kl_divergence(mu_z, log_var_z)
                )
                total_loss += loss.item() * len(xb)

        return total_loss / (S * N)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor, timestamps) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          [S, N, seq_len]
            timestamps: [S] array-like of timestamps
        Returns:
            {"load": [S, N, Q], "pv": [S, N, Q]}

        Load quantiles: from Normal(μ, σ).
        Solar quantiles: from zero-inflated Beta — gated by P(daytime).
            Q_solar(q) = 0               if q ≤ P(night)
                       = Beta.icdf(q*)   if q > P(night),  q* = (q - P(night)) / P(day)
        """
        S, N, L = x.shape
        t      = _cyclic_time_features(timestamps)                 # [S, TIME_DIM]
        t_flat = t.unsqueeze(1).expand(S, N, TIME_DIM).reshape(S * N, TIME_DIM)
        x_flat = x.reshape(S * N, L)

        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)  # [Q]

        self._model.eval()
        load_qs, pv_qs = [], []

        dataset = TensorDataset(x_flat, t_flat)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for xb, tb in loader:
                xb, tb = xb.to(self.device), tb.to(self.device)

                # Deterministic decode using posterior mean Z (no noise at inference)
                mu_z, _ = self._model.encode(xb, tb)
                mu_load, sigma_load, alpha, beta, gate_logit = self._model.decode(mu_z, tb)

                # Load: Normal quantiles  [B, Q]
                load_q = torch.distributions.Normal(
                    mu_load.unsqueeze(-1), sigma_load.unsqueeze(-1)
                ).icdf(q_tensor.unsqueeze(0))                      # [B, Q]

                # Solar: zero-inflated Beta quantiles  [B, Q]
                # torch.distributions.Beta does not implement icdf; use scipy on CPU.
                p_day   = torch.sigmoid(gate_logit)                # [B]
                p_night = 1.0 - p_day                              # [B]

                q_np     = np.array(self.quantiles, dtype=np.float64)   # [Q]
                p_day_np = p_day.cpu().numpy().astype(np.float64)        # [B]
                p_ngt_np = p_night.cpu().numpy().astype(np.float64)      # [B]
                alpha_np = alpha.cpu().numpy().astype(np.float64)        # [B]
                beta_np  = beta.cpu().numpy().astype(np.float64)         # [B]

                B_local = alpha_np.shape[0]
                pv_q_np = np.zeros((B_local, len(self.quantiles)), dtype=np.float32)
                for i in range(B_local):
                    for j, q_j in enumerate(q_np):
                        if q_j > p_ngt_np[i]:
                            q_eff_ij = float(np.clip(
                                (q_j - p_ngt_np[i]) / max(p_day_np[i], 1e-6),
                                1e-6, 1.0 - 1e-6,
                            ))
                            pv_q_np[i, j] = scipy_beta.ppf(q_eff_ij, alpha_np[i], beta_np[i])
                        # else: stays 0 (nighttime mass)

                pv_q = torch.tensor(pv_q_np) * self.pv_max        # [B, Q]

                load_qs.append(load_q.cpu())
                pv_qs.append(pv_q.cpu())

        load_pred = torch.cat(load_qs, dim=0).reshape(S, N, -1)
        pv_pred   = torch.cat(pv_qs,  dim=0).reshape(S, N, -1)
        return {"load": load_pred, "pv": pv_pred}

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        torch.save({
            "state_dict":        self._model.state_dict(),
            "latent_dim":        self.latent_dim,
            "hidden_dim":        self.hidden_dim,
            "decoder_dim":       self.decoder_dim,
            "num_layers":        self.num_layers,
            "dropout":           self.dropout,
            "quantiles":         self.quantiles,
            "n_epochs":          self.n_epochs,
            "lr":                self.lr,
            "batch_size":        self.batch_size,
            "solar_weight":      self.solar_weight,
            "gate_weight":       self.gate_weight,
            "kl_weight":         self.kl_weight,
            "daytime_threshold": self.daytime_threshold,
            "pv_eps":            self.pv_eps,
            "pv_max":            self.pv_max,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "CVAEBaseline":
        state = torch.load(path, weights_only=True)
        obj   = cls(
            latent_dim        = state["latent_dim"],
            hidden_dim        = state["hidden_dim"],
            decoder_dim       = state["decoder_dim"],
            num_layers        = state["num_layers"],
            dropout           = state["dropout"],
            quantiles         = state["quantiles"],
            n_epochs          = state["n_epochs"],
            lr                = state["lr"],
            batch_size        = state["batch_size"],
            solar_weight      = state["solar_weight"],
            gate_weight       = state["gate_weight"],
            kl_weight         = state["kl_weight"],
            daytime_threshold = state["daytime_threshold"],
            pv_eps            = state["pv_eps"],
        )
        obj.pv_max  = state["pv_max"]
        obj._model  = _CVAENet(
            latent_dim   = state["latent_dim"],
            hidden_dim   = state["hidden_dim"],
            decoder_dim  = state["decoder_dim"],
            num_layers   = state["num_layers"],
            dropout      = state["dropout"],
        ).to(obj.device)
        obj._model.load_state_dict(state["state_dict"])
        return obj
