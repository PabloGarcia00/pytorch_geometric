from pathlib import Path
from typing import Dict, Optional

import torch

from .normalization import (
    denorm_ihs,
    denorm_log1p,
    denorm_minmax,
    denorm_zscore,
    fit_ihs,
    fit_log1p,
    fit_minmax,
    fit_zscore,
    norm_ihs,
    norm_log1p,
    norm_minmax,
    norm_zscore,
)

_FIT = {
    "ihs": fit_ihs,
    "minmax": fit_minmax,
    "log1p": fit_log1p,
    "znorm": fit_zscore,
}
_NORM = {
    "ihs": norm_ihs,
    "minmax": norm_minmax,
    "log1p": norm_log1p,
    "znorm": norm_zscore,
}
_DENORM = {
    "ihs": denorm_ihs,
    "minmax": denorm_minmax,
    "log1p": denorm_log1p,
    "znorm": denorm_zscore,
}


class Transform:
    """
    Tracks normalization mode and fitted parameters per stream.
    Mode config is passed at fit time — Transform records what was used
    so inverse_transform is always consistent.

    Usage:
        t = Transform()

        # energy streams — modes from cfg dict
        scaled = t.fit_transform(cfg.earne_data.energy_norm_mode, mask=mask,
                                 load=load_raw, pv=pv_raw, net_demand=net_raw)

        # weather streams — "all_features" key applies one mode to all
        scaled_w = t.fit_transform(cfg.earne_data.weather_norm_mode,
                                   temp=temp_raw, irr=irr_raw)

        # later — inverse
        pv_raw = t.inverse_transform("pv", pv_scaled)

        # persist
        t.save(path)
        t = Transform.load(path)
    """

    def __init__(self):
        # name -> {"mode": str, "param1": Tensor, "param2": Tensor}
        self._registry: Dict[str, dict] = {}

    # ── sklearn-style API ─────────────────────────────────────────────────────

    def fit(
        self,
        mode_cfg: Dict[str, str],
        mask: Optional[torch.Tensor] = None,
        **streams: torch.Tensor,
    ) -> "Transform":
        """
        Fit scalers for each stream.
        mode_cfg: dict mapping stream name -> mode, or {"all_features": mode}
        Returns self for chaining.
        """
        all_mode = mode_cfg.get("all_features")

        for name, tensor in streams.items():
            mode = all_mode if all_mode is not None else mode_cfg.get(name)
            if mode is None:
                raise KeyError(
                    f"No normalization mode found for stream {name!r}. "
                    f"Add it to the config or use 'all_features'."
                )
            if mode not in _FIT:
                raise ValueError(
                    f"Unknown mode {mode!r}, choose from {list(_FIT)}"
                )

            data = tensor[mask.bool()] if mask is not None else tensor
            p1, p2 = _FIT[mode](data)
            self._registry[name] = {"mode": mode, "param1": p1, "param2": p2}

        return self

    def transform(self, stream: str, X: torch.Tensor) -> torch.Tensor:
        """Transform a single stream. Raises if not yet fitted."""
        entry = self._get_fitted(stream)
        p1, p2 = self._params_on(entry, X.device if torch.is_tensor(X) else "cpu")
        return _NORM[entry["mode"]](X, p1, p2)

    def fit_transform(
        self,
        mode_cfg: Dict[str, str],
        mask: Optional[torch.Tensor] = None,
        **streams: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fit and transform all streams in one call.
            scaled = t.fit_transform(cfg.earne_data.energy_norm_mode,
                                     mask=mask, load=load_raw, pv=pv_raw)
            load_scaled = scaled["load"]
        """
        self.fit(mode_cfg, mask=mask, **streams)
        return {
            name: self.transform(name, tensor)
            for name, tensor in streams.items()
        }

    def inverse_transform(self, stream: str, Y: torch.Tensor) -> torch.Tensor:
        """Invert normalization for a single stream."""
        entry = self._get_fitted(stream)
        p1, p2 = self._params_on(entry, Y.device if torch.is_tensor(Y) else "cpu")
        return _DENORM[entry["mode"]](Y, p1, p2)

    @staticmethod
    def _params_on(entry: dict, device) -> tuple:
        """Fitted params are CPU tensors (fit during process(), before any
        GPU move) -- every prior caller only ever used them post-training on
        CPU tensors, so this never surfaced. Loss-time calls need it on the
        same device as the tensor being (de)normalized, or the underlying
        torch op raises a device-mismatch error."""
        p1, p2 = entry["param1"], entry["param2"]
        return p1.to(device), p2.to(device)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        torch.save(self._registry, path)

    @classmethod
    def load(cls, path: Path) -> "Transform":
        instance = cls()
        instance._registry = torch.load(path, weights_only=True)
        return instance

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_fitted(self, stream: str) -> dict:
        if stream not in self._registry:
            raise RuntimeError(
                f"Stream {stream!r} has not been fitted. "
                f"Fitted streams: {list(self._registry)}"
            )
        return self._registry[stream]

    def is_fitted(self, stream: str) -> bool:
        return stream in self._registry

    def __repr__(self) -> str:
        lines = ["Transform("]
        if not self._registry:
            lines.append("  <no streams fitted>")
        for name, entry in self._registry.items():
            lines.append(
                f"  {name:<16} mode={entry['mode']:<8} "
                f"p1={entry['param1']:.4f}  p2={entry['param2']:.4f}"
            )
        lines.append(")")
        return "\n".join(lines)
