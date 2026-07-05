import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head


@register_head("earne_quantile")
class EARNeQuantileHead(nn.Module):
    r"""
    Custom head for EARNe that outputs dual quantiles for Load and PV.
    Expects node embeddings (dim_in) as input.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Use values from our custom config
        self.n_quantiles = cfg.model.n_quantiles

        # Two Output Heads: One for Load, one for PV
        # Each outputs [Batch * Nodes, n_quantiles]
        self.head_load = nn.Linear(dim_in, self.n_quantiles)
        self.head_pv = nn.Linear(dim_in, self.n_quantiles)

    def forward(self, batch):
        # batch.x contains the embeddings from the GNN body
        x = batch.x

        # Predict both components
        q_load = self.head_load(x)
        q_pv = self.head_pv(x)

        # Attach predictions to the batch object for the loss function
        # batch.q_load = q_load
        # batch.q_pv = q_pv

        # Concatenate into [N, 2 * n_quantiles] for standard GraphGym flow
        return torch.cat([q_load, q_pv], dim=1)
