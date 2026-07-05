"""Historical mean baseline: per-(hour_of_day × day_of_week) quantile lookup."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class HistoricalMeanBaseline:
    """
    For each (hour_of_day, day_of_week) bin, fit quantiles of load and PV
    from the training set. At inference, look up the bin for each target timestamp.
    """

    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        # {(hour, dow): {"load": [N, Q], "pv": [N, Q]}}
        self._table: dict = {}

    def fit(
        self,
        x_train: torch.Tensor,
        y_load_train: torch.Tensor,
        y_pv_train: torch.Tensor,
        timestamps_train: List,
    ):
        """
        Args:
            x_train:          [samples, N, seq_len] — unused (kept for API symmetry)
            y_load_train:     [samples, N]
            y_pv_train:       [samples, N]
            timestamps_train: list of target timestamps, length = samples
        """
        ts = pd.to_datetime(timestamps_train)
        load = y_load_train.numpy()  # [samples, N]
        pv = y_pv_train.numpy()

        keys = list(zip(ts.hour.tolist(), ts.dayofweek.tolist()))
        unique_keys = set(keys)

        for key in unique_keys:
            idx = [i for i, k in enumerate(keys) if k == key]
            load_bin = load[idx]  # [bin_size, N]
            pv_bin = pv[idx]
            self._table[key] = {
                "load": np.quantile(load_bin, self.quantiles, axis=0).T,  # [N, Q]
                "pv": np.quantile(pv_bin, self.quantiles, axis=0).T,
            }

        return self

    def predict(
        self, x: torch.Tensor, timestamps: List
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          [samples, N, seq_len] (unused)
            timestamps: list of target timestamps, length = samples
        Returns:
            dict with "load" and "pv", each [samples, N, Q]
        """
        ts = pd.to_datetime(timestamps)
        n_nodes = x.shape[1]
        q = len(self.quantiles)
        load_preds = np.zeros((len(timestamps), n_nodes, q), dtype=np.float32)
        pv_preds = np.zeros_like(load_preds)

        global_load = np.zeros((n_nodes, q), dtype=np.float32)
        global_pv = np.zeros((n_nodes, q), dtype=np.float32)
        if self._table:
            global_load = np.mean(
                [v["load"] for v in self._table.values()], axis=0
            )
            global_pv = np.mean(
                [v["pv"] for v in self._table.values()], axis=0
            )

        for i, (h, d) in enumerate(zip(ts.hour.tolist(), ts.dayofweek.tolist())):
            entry = self._table.get((h, d))
            if entry is None:
                load_preds[i] = global_load
                pv_preds[i] = global_pv
            else:
                load_preds[i] = entry["load"]
                pv_preds[i] = entry["pv"]

        return {
            "load": torch.tensor(load_preds),
            "pv": torch.tensor(pv_preds),
        }

    def save(self, path: Path):
        torch.save({"quantiles": self.quantiles, "table": self._table}, path)

    @classmethod
    def load(cls, path: Path) -> "HistoricalMeanBaseline":
        state = torch.load(path, weights_only=False)
        obj = cls(quantiles=state["quantiles"])
        obj._table = state["table"]
        return obj
