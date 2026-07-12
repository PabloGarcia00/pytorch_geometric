import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("baseline_lstm_temporal")
class LSTMTemporalEncoder(nn.Module):
    """
    Bidirectional LSTM trunk (ported from baselines/BiLSTM.py's _BiLSTM)
    wired into EARNeNetwork's encoder slot. Runs the [T, C] window (net
    demand + operational flag [+ weather]) through a BiLSTM, takes the
    final timestep's hidden state, fuses in calendar features, and
    projects to emb_dim -- matches EARNeTemporalEncoder's [N, emb_dim]
    output contract so earne_network.py/earne_quantile/earne_loss need no
    changes to consume this encoder.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.seq_len = cfg.model.seq_len
        self.weather_mode = cfg.earne_data.weather_mode
        C_x = cfg.model.dim_in + 1  # +1 for operational flag
        self.n_weather = (
            len(cfg.earne_data.weather_features) if self.weather_mode else 0
        )
        input_size = C_x + self.n_weather

        hidden_dim = cfg.baseline.lstm_hidden_dim
        num_layers = cfg.baseline.lstm_layers
        dropout = cfg.baseline.lstm_dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def _calendar(self, batch):
        # See baseline_mlp_encoder.MLPTemporalEncoder._calendar for the
        # broadcast/last-step rationale.
        N = batch.x.shape[0]
        B = batch.num_graphs
        T = self.seq_len
        temporal = batch.temporal.view(B, T, 6)
        nodes_per_graph = N // B
        temporal = (
            temporal.unsqueeze(1)
            .expand(B, nodes_per_graph, T, 6)
            .reshape(N, T, 6)
        )
        return temporal[:, -1, :]  # [N, 6]

    def forward(self, batch):
        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_x]
        if self.weather_mode:
            x_in = torch.cat([x_in, batch.weather], dim=-1)

        out, _ = self.lstm(x_in)
        last = out[:, -1, :]  # [N, hidden_dim * 2]

        t = self._calendar(batch)
        batch.x = self.integration(torch.cat([last, t], dim=-1))
        return batch
