"""Persistence baseline: predict next = last observed value."""

from pathlib import Path
from typing import Dict, List

import torch


class PersistenceBaseline:
    """
    Naïve persistence: ŷ(t+1) = x(t) for every node.
    All quantiles receive the same point estimate.
    """

    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def fit(self, x_train, y_load_train, y_pv_train):
        # stateless — nothing to fit
        return self

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [samples, N, seq_len] net demand history (scaled)
        Returns:
            dict with "load" and "pv", each [samples, N, Q]
        """
        last_step = x[:, :, -1]  # [samples, N]
        q = len(self.quantiles)
        pred = last_step.unsqueeze(-1).expand(-1, -1, q)  # [samples, N, Q]
        return {"load": pred, "pv": pred}

    def save(self, path: Path):
        torch.save({"quantiles": self.quantiles}, path)

    @classmethod
    def load(cls, path: Path) -> "PersistenceBaseline":
        state = torch.load(path, weights_only=True)
        return cls(quantiles=state["quantiles"])
