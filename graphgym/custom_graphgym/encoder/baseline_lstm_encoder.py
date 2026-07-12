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
    final timestep's hidden state, fuses in a calendar-time embedding (same
    time_encoder CNN stream as EARNeTemporalEncoder, run over the full
    window), and projects to emb_dim -- matches EARNeTemporalEncoder's
    [N, emb_dim] output contract so earne_network.py/earne_quantile/
    earne_loss need no changes to consume this encoder.
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

        # Time stream: identical structure to EARNeTemporalEncoder's
        # time_encoder (stream 3) -- CNN over the full [T, 6] cyclic
        # calendar window, output width emb_dim // 4 in the same ratio.
        time_emb_dim = emb_dim // 4
        self.time_encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, dilation=1, padding="same"),
            nn.GELU(),
            nn.Conv1d(16, time_emb_dim, kernel_size=3, dilation=4, padding="same"),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def _time_window(self, batch):
        # batch.temporal is [B*T, 6] (one [T, 6] block per graph); broadcast
        # each graph's calendar window out to its N nodes.
        N = batch.x.shape[0]
        B = batch.num_graphs
        T = self.seq_len
        temporal = batch.temporal.view(B, T, 6)
        nodes_per_graph = N // B
        return (
            temporal.unsqueeze(1)
            .expand(B, nodes_per_graph, T, 6)
            .reshape(N, T, 6)
        )  # [N, T, 6]

    def forward(self, batch):
        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_x]
        if self.weather_mode:
            x_in = torch.cat([x_in, batch.weather], dim=-1)

        out, _ = self.lstm(x_in)
        last = out[:, -1, :]  # [N, hidden_dim * 2]

        temporal = self._time_window(batch)  # [N, T, 6]
        z_time = self.time_encoder(temporal.permute(0, 2, 1))  # [N, emb_dim // 4]

        batch.x = self.integration(torch.cat([last, z_time], dim=-1))
        return batch
