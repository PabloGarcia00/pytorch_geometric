import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network 
from torch_geometric.graphgym.config import cfg

@register_network('earne_network')
class EARNeNetwork(nn.Module):
    """
    EARNe Network architecture:
    1. Temporal Encoder: (e.g., 1D-CNN) - defined by cfg.model.node_encoder_name
    2. Spatial Body: (e.g., GCN, GAT) - defined by cfg.gnn.layer_type
    3. Quantile Head: (Outputs quantiles) - defined by cfg.model.head_name
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        
        hidden_channels = cfg.model.hidden_channels
        self.layer_type = cfg.gnn.layer_type
        
        # 1. Temporal Node Encoder (e.g., 'earne_temporal')
        Encoder = register.node_encoder_dict[cfg.model.node_encoder_name]
        self.encoder = Encoder(hidden_channels) 
        
        # 2. Spatial GNN Body (Flexible MP Layers)
        self.convs = nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.convs.append(self.build_conv(hidden_channels, hidden_channels))
        
        # 3. Quantile Prediction Head
        Head = register.head_dict[cfg.model.head_name]
        self.head = Head(dim_in=hidden_channels, dim_out=dim_out)

    def build_conv(self, dim_in, dim_out):
        """Standard GNN factory for benchmarks."""
        if self.layer_type == 'gcnconv':
            return pyg_nn.GCNConv(dim_in, dim_out)
        elif self.layer_type == 'gatconv':
            heads = cfg.gnn.att_heads
            dropout = cfg.gnn.dropout

            if dim_out % heads != 0:
                raise ValueError(f"{dim_out} must be divisible by number of heads")
            return pyg_nn.GATConv(
                    dim_in,
                    out_channels=64//heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                    )

        elif self.layer_type == 'sageconv':
            return pyg_nn.SAGEConv(dim_in, dim_out)
        elif self.layer_type == 'ginconv':
            # GIN requires an internal MLP
            nn1 = nn.Sequential(
                nn.Linear(dim_in, dim_out), 
                nn.ReLU(), 
                nn.Linear(dim_out, dim_out)
            )
            return pyg_nn.GINConv(nn1)
        else:
            raise ValueError(f'Spatial layer type "{self.layer_type}" not supported.')

    def forward(self, batch):
        # Cache batch for custom loss functions (like physics loss)
        register.batch = batch

        # Step 1: Temporal Encoding (e.g., [N, T, 1] -> [N, D])
        batch = self.encoder(batch)
        
        # Step 2: Spatial Message Passing (Graph structure)
        x, edge_index = batch.x, batch.edge_index
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=cfg.gnn.dropout, training=self.training)
        
        batch.x = x
        
        # Step 3: Quantile Head (Quantile projections)
        pred = self.head(batch)
        
        # Ground truths + Mask for regression task
        true = torch.stack([batch.y_load, batch.y_pv, batch.mask], dim=1)
        
        return pred, true
