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
    1. Temporal Encoder: (1D-CNN) Processes [T, 1] Net Demand History
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
        
        # 2. Integration Layer: Projects [Temporal_Emb + Conditions] -> hidden_channels
        # Temporal Embedding (hidden_channels) + Weather + Temporal (6 cyclical feats)
        weather_dim = len(cfg.earne_data.weather_features)
        temporal_dim = 6 # Month, Weekday, Hour (sin/cos)
        integration_in = hidden_channels + weather_dim + temporal_dim
        self.integration = nn.Linear(integration_in, hidden_channels)
        
        # 3. Spatial GNN Body (Flexible MP Layers)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.convs.append(self.build_conv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # 4. Quantile Prediction Head
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
                    out_channels=dim_out // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                    )
        elif self.layer_type == 'sageconv':
            return pyg_nn.SAGEConv(dim_in, dim_out)
        else:
            raise ValueError(f'Spatial layer type "{self.layer_type}" not supported.')

    def forward(self, batch):
        # Step 1: Temporal Encoding (e.g., [N, 96, 1] -> [N, D])
        batch = self.encoder(batch)
        
        # Step 2: Integrate Conditions (Weather + Time)
        # batch.weather shape: [N, W_Feats + T_Feats]
        x_cond = batch.weather
        x = torch.cat([batch.x, x_cond], dim=1)
        x = F.relu(self.integration(x))
        
        # Step 3: Spatial Message Passing (Graph structure)
        edge_index = batch.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = x + x_in # Residual connection
            x = norm(x)
            x = F.dropout(x, p=cfg.gnn.dropout, training=self.training)
        
        batch.x = x
        
        # Step 4: Quantile Head (Quantile projections)
        pred = self.head(batch)
        
        # Cache batch for custom loss functions (like physics loss)
        # Done after head() so q_load/q_pv are attached
        register.batch = batch

        # Ground truths + Mask for regression task
        true = torch.stack([batch.y_load, batch.y_pv, batch.mask], dim=1)
        
        return pred, true
