import torch
import torch.nn as nn
from typing import Tuple, List
from ..layer.gnn import SpectralGraphConvolution, MultiHeadAttention
from ..layer.rnn import PeepholeLSTMCell

class SGCAPL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 8, n_heads: int = 3):
        super().__init__()
        self.n_layers = n_layers

        self.sgc = SpectralGraphConvolution(input_dim, hidden_dim, K=1)
        self.attention = MultiHeadAttention(hidden_dim, hidden_dim, n_heads)
        self.lstm_cells = nn.ModuleList([
            PeepholeLSTMCell(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor,
                h_states: List[torch.Tensor] = None,
                c_states: List[torch.Tensor] = None) -> Tuple[torch.Tensor, List, List]:
        N = nodes.shape[0]
        device = nodes.device

        # Initialize states if not provided
        if h_states is None:
            h_states = [torch.zeros(N, self.lstm_cells[0].hidden_size, device=device)
                        for _ in range(self.n_layers)]
        if c_states is None:
            c_states = [torch.zeros(N, self.lstm_cells[0].hidden_size, device=device)
                        for _ in range(self.n_layers)]

        x = self.sgc(nodes, adjacency)
        x = self.attention(x, adjacency)

        new_h_states = []
        new_c_states = []

        for layer in range(self.n_layers):
            c_lower = c_states[layer - 1] if layer > 0 else None
            h, c = self.lstm_cells[layer](x, h_states[layer], c_states[layer], c_lower)
            new_h_states.append(h)
            new_c_states.append(c)
            x = h  # Use hidden state as input to next layer

        output = self.output_proj(x)
        return output, new_h_states, new_c_states
