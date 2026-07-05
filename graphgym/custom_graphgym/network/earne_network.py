import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network


@register_network("earne_network")
class EARNeNetwork(nn.Module):
    """
    EARNe Network architecture:
    1. Temporal Encoder: (1D-CNN) Processes [T, dim_in] history (1 channel for
       net demand, 2 channels for import/export when dim_in=2)
    2. Integration: Concatenates Temporal Embedding with [Weather + Time] Conditions
    3. Spatial Body: (GCN, GAT, etc.) Message passing on the graph
    4. Quantile Head: Outputs Load and PV quantiles
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        hidden_channels = cfg.model.hidden_channels
        self.layer_type = cfg.gnn.layer_type

        # 1. Temporal Node Encoder (Net Demand history)
        Encoder = register.node_encoder_dict[cfg.model.node_encoder_name]
        self.encoder = Encoder(hidden_channels)

        # 2. Spatial GNN Body (Flexible MP Layers)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.convs.append(self.build_conv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # 4. Quantile Prediction Head
        Head = register.head_dict[cfg.model.head_name]
        self.head = Head(dim_in=hidden_channels, dim_out=dim_out)

        # 5. Optional Learnable Graph
        self.graph_mode = cfg.earne_data.get("graph_mode", "spatial_knn")
        if self.graph_mode in ("learned_corr", "static_corr"):
            from custom_graphgym.loader.graph_builder import GraphBuilder

            lambda_init = cfg.earne_data.get("lambda_graph", 0.25)
            dual_read = cfg.model.dim_in == 2
            dual_agg = cfg.earne_data.get("dual_agg", "max")
            self.graph_builder = GraphBuilder(
                lambda_threshold=lambda_init,
                dual_read=dual_read,
                dual_agg=dual_agg,
            )
            self.lambda_threshold = nn.Parameter(torch.tensor(lambda_init))

        self.dropout = cfg.gnn.dropout

    def build_conv(self, dim_in, dim_out):
        """Standard GNN factory for benchmarks."""
        if self.layer_type == "gcnconv":
            return pyg_nn.GCNConv(dim_in, dim_out)
        elif self.layer_type == "gatconv":
            heads = cfg.gnn.att_heads
            dropout = cfg.gnn.dropout
            edge_dim = 1
            if dim_out % heads != 0:
                raise ValueError(
                    f"{dim_out} must be divisible by number of heads"
                )
            return pyg_nn.GATv2Conv(
                dim_in,
                out_channels=dim_out // heads,
                heads=heads,
                dropout=dropout,
                concat=True,
                edge_dim=edge_dim,
            )

        elif self.layer_type == "sageconv":
            return pyg_nn.SAGEConv(dim_in, dim_out)
        else:
            raise ValueError(
                f'Spatial layer type "{self.layer_type}" not supported.'
            )

    def forward(self, batch):
        raw_x, second_stream = self._extract_raw_x(batch)
        batch = self.encoder(batch)
        x = batch.x

        edge_index, edge_attr = self._get_graph(batch, raw_x, second_stream)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = norm(x)
            if self.layer_type == "gatconv":
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        batch.x = x
        pred = self.head(batch)
        true = torch.stack(
            [batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1
        )
        return pred, true

    def _extract_raw_x(self, batch):
        """
        Single read: batch.x [N, T, 1] → (net_demand [N, T], None)
        Dual read:   batch.x [N, T, 2] → (consumption [N, T], generation [N, T])
        """
        if batch.x.shape[-1] == 2:
            return batch.x[:, :, 0], batch.x[:, :, 1]
        return batch.x[:, :, 0], None

    def _get_graph(self, batch, raw_x, second_stream=None):
        if self.graph_mode == "learned_corr":
            # build_graph expects a single [N, T] signal — use net demand in both modes
            # for the rolling window correlation; dual stream awareness lives in static_corr
            edge_index, edge_attr = self.graph_builder.build_graph(
                raw_x if second_stream is None else raw_x - second_stream,
                lambda_threshold=self.lambda_threshold,
                batch_vec=batch.batch,
            )
            return edge_index, edge_attr

        elif self.graph_mode == "static_corr":
            # abs_corr was precomputed with full dual-stream awareness in the dataset
            return self.graph_builder.threshold_graph(
                batch.abs_corr,
                lambda_threshold=self.lambda_threshold,
            )

        return batch.edge_index, getattr(batch, "edge_attr", None)
