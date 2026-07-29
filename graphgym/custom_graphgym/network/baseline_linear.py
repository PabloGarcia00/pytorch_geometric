import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets
from ._baseline_common import (
    PerNodeLinearQuantileHead,
    baseline_current_in_features,
    baseline_in_features,
    build_current_features,
    build_features,
    node_ids_for_batch,
    stack_true,
)


@register_network("baseline_linear")
class BaselineLinearNetwork(nn.Module):
    """
    Per-node linear quantile regression baseline.

    One independent linear map per household (household i never sees
    household j's data, at fit time or inference time -- see
    PerNodeLinearQuantileHead) straight to quantiles for whichever of
    load/PV are selected via cfg.model.predict_targets (default: PV only),
    trained with the same masked pinball loss as earne_network
    (earne_loss.py) via the standard GraphGym/Lightning training loop -- no
    gradient/training-loop changes needed, this is a fully standard
    nn.Module.

    cfg.baseline.linear_feature_mode selects the input: 'window' (default)
    flattens a [seq_len, C] history window (+ calendar anchor); 'current'
    uses only the most recent timestep's raw signal (+ calendar anchor) --
    see results_archives/ for the write-up motivating the latter.

    Bypasses earne_network's encoder/GNN-body/head chain entirely (like
    baseline_cvae_network.py / st_sgc_caps.py) so per-node fitting is
    structurally guaranteed: no message passing, no cross-household mixing.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        num_nodes = cfg.share.num_nodes
        self.weather_mode = cfg.earne_data.weather_mode
        n_weather = (
            len(cfg.earne_data.weather_features) if self.weather_mode else 0
        )
        self.feature_mode = cfg.baseline.linear_feature_mode
        assert self.feature_mode in ("window", "current"), (
            f"Unknown linear_feature_mode: {self.feature_mode!r}"
        )
        if self.feature_mode == "window":
            in_features = baseline_in_features(
                cfg.model.dim_in, self.weather_mode, n_weather
            )
        else:
            in_features = baseline_current_in_features(
                cfg.model.dim_in, self.weather_mode, n_weather
            )
        # One quantile block per selected target (cfg.model.predict_targets,
        # default PV only), in active_targets() order -- earne_loss slices
        # pred the same way, so the two agree without the head needing to
        # know target names itself.
        out_features = len(active_targets()) * cfg.model.n_quantiles

        self.num_nodes = num_nodes
        self.head = PerNodeLinearQuantileHead(num_nodes, in_features, out_features)

    def forward(self, batch):
        if self.feature_mode == "window":
            x = build_features(batch, self.weather_mode, self.num_nodes)
        else:
            x = build_current_features(batch, self.weather_mode, self.num_nodes)
        node_ids = node_ids_for_batch(batch, self.num_nodes)
        pred = self.head(x, node_ids)  # [N, n_quantiles * len(targets)]
        true = stack_true(batch)
        return pred, true
