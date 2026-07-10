import torch
import torch.nn as nn
from typing import List, Tuple


class EdgeDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = (128, 256, 1024, 512)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        device = z.device
        triu = torch.triu_indices(N, N, offset=1, device=device)
        pair_features = torch.cat([z[triu[0]], z[triu[1]]], dim=-1)
        edge_weights = self.decoder(pair_features).squeeze(-1)
        edges = torch.zeros(N, N, device=device)
        edges[triu[0], triu[1]] = torch.sigmoid(edge_weights)
        return edges + edges.T


class NodeDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dims: List[int] = (512, 256, 1024, 512, 256)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class GenerativeDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_hidden: List[int] = (128, 256, 1024, 512),
        node_hidden: List[int] = (512, 256, 1024, 512, 256),
        output_dim: int = 1,
    ):
        super().__init__()
        self.edge_decoder = EdgeDecoder(input_dim * 2, edge_hidden)
        self.node_decoder = NodeDecoder(input_dim, output_dim, node_hidden)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edge_decoder(z), self.node_decoder(z)
