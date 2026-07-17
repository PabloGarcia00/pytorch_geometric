import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head

from ..target_utils import active_targets


@register_head("earne_quantile")
class EARNeQuantileHead(nn.Module):
    r"""
    Custom head for EARNe that outputs quantiles for whichever of Load/PV
    are selected via cfg.model.predict_targets (default: PV only).
    Expects node embeddings (dim_in) as input.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Use values from our custom config
        self.n_quantiles = cfg.model.n_quantiles
        self.targets = active_targets()

        # One output head per selected target, each [Batch * Nodes, n_quantiles]
        self.heads = nn.ModuleDict(
            {t: nn.Linear(dim_in, self.n_quantiles) for t in self.targets}
        )

    def forward(self, batch):
        # batch.x contains the embeddings from the GNN body
        x = batch.x

        # Concatenate into [N, n_quantiles * len(targets)], load before pv
        # when both are selected (see active_targets()).
        return torch.cat([self.heads[t](x) for t in self.targets], dim=1)
