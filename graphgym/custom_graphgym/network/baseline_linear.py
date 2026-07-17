import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets
from ._baseline_common import (
    PerNodeLinearQuantileHead,
    baseline_in_features,
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
    PerNodeLinearQuantileHead) from a flattened [seq_len, C] history window
    (+ calendar anchor) straight to quantiles for whichever of load/PV are
    selected via cfg.model.predict_targets (default: PV only), trained
    with the same masked pinball loss as earne_network (earne_loss.py) via
    the standard GraphGym/Lightning training loop -- no gradient/training-
    loop changes needed, this is a fully standard nn.Module.

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
        in_features = baseline_in_features(
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
        x = build_features(batch, self.weather_mode, self.num_nodes)
        node_ids = node_ids_for_batch(batch, self.num_nodes)
        pred = self.head(x, node_ids)  # [N, n_quantiles * len(targets)]
        true = stack_true(batch)
        return pred, true
