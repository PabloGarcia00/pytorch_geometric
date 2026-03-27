import torch
import torch.nn as nn
from typing import List, Tuple

class EdgeDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 256, 1024, 512]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        device = z.device
        triu_indices = torch.triu_indices(N, N, offset=1, device=device)
        row_indices, col_indices = triu_indices[0], triu_indices[1]
        z_i = z[row_indices]
        z_j = z[col_indices]
        pair_features = torch.cat([z_i, z_j], dim=-1)
        edge_weights = self.decoder(pair_features).squeeze()
        edges = torch.zeros(N, N, device=device)
        edges[row_indices, col_indices] = torch.sigmoid(edge_weights)
        edges = edges + edges.T
        return edges


class NodeDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dims: List[int] = [512, 256, 1024, 512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class GenerativeDecoder(nn.Module):
    def __init__(self, input_dim: int,
                 edge_hidden: List[int] = [128, 256, 1024, 512],
                 node_hidden: List[int] = [512, 256, 1024, 512, 256]):
        super().__init__()
        self.edge_decoder = EdgeDecoder(input_dim * 2, edge_hidden)
        self.node_decoder = NodeDecoder(input_dim, 1, node_hidden)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = self.edge_decoder(z)
        nodes = self.node_decoder(z)
        return edges, nodes
