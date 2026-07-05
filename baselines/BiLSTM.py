"""BiLSTM baseline for solar/load disaggregation: bidirectional LSTM with pinball loss."""

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    pred:   [batch, Q]
    target: [batch]
    """
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    errors = target.unsqueeze(-1) - pred  # [batch, Q]
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


class _BiLSTM(nn.Module):
    def __init__(self, hidden_dim: int, n_quantiles: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, n_quantiles * 2)  # load + pv
        self.n_quantiles = n_quantiles

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [batch, seq_len] -> [batch, seq_len, 1]
        x_unsqueezed = x.unsqueeze(-1)
        out, _ = self.lstm(x_unsqueezed)  # out: [batch, seq_len, hidden_dim * 2]
        last_out = out[:, -1, :]  # [batch, hidden_dim * 2]
        pred = self.fc(last_out)  # [batch, Q*2]
        load_pred = pred[:, : self.n_quantiles]
        pv_pred = pred[:, self.n_quantiles :]
        return load_pred, pv_pred


class BiLSTMBaseline:
    """
    2-layer Bidirectional LSTM trained per-node independently with pinball loss.
    Input: seq_len net demand values. Output: Q quantiles for load and PV.
    """

    def __init__(
        self,
        seq_len: int = 96,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: List[float] = None,
        n_epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 4096,
        device: str = "auto",
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self._model: _BiLSTM | None = None

    def fit(
        self,
        x_train: torch.Tensor,
        y_load_train: torch.Tensor,
        y_pv_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_load_val: torch.Tensor | None = None,
        y_pv_val: torch.Tensor | None = None,
    ):
        """
        Args:
            x_train:      [samples, N, seq_len]
            y_load_train: [samples, N]
            y_pv_train:   [samples, N]
        """
        S, N, L = x_train.shape
        # Flatten nodes into batch dimension: [samples*N, seq_len]
        x_flat = x_train.reshape(S * N, L)
        load_flat = y_load_train.reshape(S * N)
        pv_flat = y_pv_train.reshape(S * N)

        self._model = _BiLSTM(
            hidden_dim=self.hidden_dim,
            n_quantiles=len(self.quantiles),
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        dataset = TensorDataset(x_flat, load_flat, pv_flat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.n_epochs + 1):
            self._model.train()
            total_loss = 0.0
            for xb, lb, pb in loader:
                xb, lb, pb = xb.to(self.device), lb.to(self.device), pb.to(self.device)
                load_pred, pv_pred = self._model(xb)
                loss = _pinball_loss(load_pred, lb, self.quantiles) + _pinball_loss(pv_pred, pb, self.quantiles)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / (S * N)

            val_str = ""
            if x_val is not None:
                val_loss = self._eval_loss(x_val, y_load_val, y_pv_val)
                val_str = f"  val_loss={val_loss:.4f}"

            print(f"epoch {epoch:3d}/{self.n_epochs}  train_loss={avg_loss:.4f}{val_str}")

        return self

    def _eval_loss(self, x_val, y_load_val, y_pv_val) -> float:
        S, N, L = x_val.shape
        x_flat = x_val.reshape(S * N, L)
        load_flat = y_load_val.reshape(S * N)
        pv_flat = y_pv_val.reshape(S * N)
        
        self._model.eval()
        total_loss = 0.0
        
        dataset = TensorDataset(x_flat, load_flat, pv_flat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for xb, lb, pb in loader:
                xb, lb, pb = xb.to(self.device), lb.to(self.device), pb.to(self.device)
                load_pred, pv_pred = self._model(xb)
                loss = _pinball_loss(load_pred, lb, self.quantiles) + _pinball_loss(pv_pred, pb, self.quantiles)
                total_loss += loss.item() * len(xb)
                
        return total_loss / (S * N)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [samples, N, seq_len]
        Returns:
            dict with "load" and "pv", each [samples, N, Q]
        """
        S, N, L = x.shape
        x_flat = x.reshape(S * N, L)
        
        self._model.eval()
        load_preds = []
        pv_preds = []
        
        dataset = TensorDataset(x_flat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                lp, pp = self._model(xb)
                load_preds.append(lp.cpu())
                pv_preds.append(pp.cpu())
                
        load_pred = torch.cat(load_preds, dim=0)
        pv_pred = torch.cat(pv_preds, dim=0)
        
        return {
            "load": load_pred.reshape(S, N, -1),
            "pv": pv_pred.reshape(S, N, -1),
        }

    def save(self, path: Path):
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "quantiles": self.quantiles,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BiLSTMBaseline":
        state = torch.load(path, weights_only=True)
        obj = cls(
            seq_len=state["seq_len"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            quantiles=state["quantiles"],
            n_epochs=state["n_epochs"],
            lr=state["lr"],
            batch_size=state["batch_size"],
        )
        obj._model = _BiLSTM(
            hidden_dim=state["hidden_dim"],
            n_quantiles=len(state["quantiles"]),
            num_layers=state["num_layers"],
            dropout=state["dropout"]
        ).to(obj.device)
        obj._model.load_state_dict(state["state_dict"])
        return obj
