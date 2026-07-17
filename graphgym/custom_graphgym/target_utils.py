"""Shared helper for cfg.model.predict_targets -- which of load/PV each
model predicts. Every network/head/loss/metric that builds or reads
per-target output blocks (earne_head.py, earne_loss.py, baseline_linear.py,
baseline_svr.py, baseline_knn.py, baseline_cvae_network.py, cvae_loss.py,
st_sgc_caps.py, st_caps_loss.py, st_caps_metric.py,
custom_graphgym/metric/regression.py, notebooks/calculate_metrics.py) goes
through active_targets() so column order agrees everywhere, regardless of
the order the user lists targets in config.

Not a @register_config/@register_* module -- a plain utility imported
directly, same as custom_graphgym/transform/transform.py's Transform class.
"""

from torch_geometric.graphgym.config import cfg

ALL_TARGETS = ("load", "pv")


def active_targets():
    """Canonical, order-stable subset of ALL_TARGETS selected via
    cfg.model.predict_targets. Always returns "load" before "pv" when both
    are selected, independent of the order they appear in config.
    """
    selected = set(cfg.model.predict_targets)
    unknown = selected - set(ALL_TARGETS)
    if unknown:
        raise ValueError(
            f"cfg.model.predict_targets has unknown target(s): {sorted(unknown)}; "
            f"expected a subset of {ALL_TARGETS}"
        )
    if not selected:
        raise ValueError(
            "cfg.model.predict_targets must select at least one of "
            f"{ALL_TARGETS}, got an empty list"
        )
    return [t for t in ALL_TARGETS if t in selected]
