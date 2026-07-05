import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("earne_temporal")
class EARNeTemporalEncoder(nn.Module):
    """
    Three-stream parallel encoder for EARNe.
    Separate dilated 1D CNNs for net demand (4 layers), weather (grouped pair-wise
    then cross-feature), and time encoding (full sequence CNN).
    Weather gates the net demand embedding before integration to give the model
    explicit solar-context for disaggregation.
    Outputs [N, emb_dim] for the GNN body.
    """

    def __init__(self, emb_dim):
        super().__init__()
        # after
        C_x = cfg.model.dim_in + 1  # +1 for operational flag

        self.weather_mode = cfg.earne_data.weather_mode
        self.seq_len = cfg.model.seq_len

        # Stream 1: net demand CNN
        # Dilations 1, 4, 8, 16 cover local spikes → 15min → 1hr → 2hr → 4hr patterns
        self.net_encoder = nn.Sequential(
            nn.Conv1d(C_x, 32, kernel_size=3, dilation=1, padding="same"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=4, padding="same"),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, dilation=8, padding="same"),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=16, padding="same"),
            nn.GELU(),  # fix: was missing after 4th conv
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, emb_dim),
            nn.GELU(),
        )

        # Stream 2: weather CNN
        # groups=6 in first conv keeps (mean, std) pairs coupled before cross-feature mixing
        # Only built when weather_mode=True (fix: was always built)
        if self.weather_mode:
            weather_cols = cfg.earne_data.weather_features
            mean_cols = len([f for f in weather_cols if not f.endswith("_std")])
            std_cols = len([f for f in weather_cols if f.endswith("_std")])
            assert (
                mean_cols > 0 or std_cols > 0
            ), "at least one feature required"

            self.W = len(weather_cols)

            self.n_pairs = self.W // 2 if mean_cols == std_cols else None

            self.weather_encoder = nn.Sequential(
                nn.Conv1d(
                    self.W,
                    self.W * 2,
                    kernel_size=3,
                    dilation=1,
                    padding="same",
                    groups=self.n_pairs if self.n_pairs else 1,
                ),
                nn.GELU(),
                nn.Conv1d(
                    self.W * 2,
                    self.W * 4,
                    kernel_size=3,
                    dilation=4,
                    padding="same",
                ),  # cross-feature
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.W * 4, emb_dim // 2),
                nn.GELU(),
            )
            # Weather gate: modulates net demand embedding with solar/weather context
            # Key for disaggregation: "when irradiance is high, amplify PV-related features"
            self.weather_gate = nn.Linear(emb_dim // 2, emb_dim)

        # Stream 3: time encoding CNN over full sequence
        # Using full T instead of last timestep only — solar arc shape across window
        # is the primary disaggregation signal (PV ramp-up/down pattern)
        self.time_encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, dilation=1, padding="same"),
            nn.GELU(),
            nn.Conv1d(
                16, emb_dim // 4, kernel_size=3, dilation=4, padding="same"
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Integration: fuse all streams → [N, emb_dim]
        d_concat = (
            emb_dim + (emb_dim // 2 if self.weather_mode else 0) + emb_dim // 4
        )
        self.integration = nn.Sequential(
            nn.Linear(d_concat, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def forward(self, batch):

        # Stream 3: time encoding [N, 6, T] → [N, emb_dim//4]
        N = batch.x.shape[0]
        B = batch.num_graphs
        T = self.seq_len

        # reshape to [B, T, 6] then repeat for each node in each graph
        temporal = batch.temporal.view(B, T, 6)  # [B, T, 6]
        nodes_per_graph = N // B
        temporal = (
            temporal.unsqueeze(1)
            .expand(B, nodes_per_graph, T, 6)
            .reshape(N, T, 6)
        )  # [N, T, 6]
        z_time = self.time_encoder(temporal.permute(0, 2, 1))

        # Stream 1: net demand [N, C_x, T] → [N, emb_dim]

        x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_x]
        z_net = self.net_encoder(x_in.permute(0, 2, 1))

        if self.weather_mode:
            w = batch.weather.permute(0, 2, 1)  # [N, N_features, T]
            if self.n_pairs:
                # Interleave (mean, std) pairs before grouped conv
                # fix: ensures groups=6 conv operates on correct pairs regardless
                # of whether dataset stores [mean×6, std×6] or already interleaved
                means = w[:, : self.n_pairs, :]  # [N, 6, T]
                stds = w[:, self.n_pairs :, :]  # [N, 6, T]
                w_input = torch.stack([means, stds], dim=2).reshape(
                    w.shape[0], self.W, w.shape[2]
                )  # [N, N_features, T] interleaved
            else:
                # just pass the feature
                w_input = w

            # Stream 2: weather [N, N_features, T] → [N, emb_dim//2]
            z_w = self.weather_encoder(w_input)

            # Weather gate: use weather context to modulate net demand embedding
            # Gives model explicit signal: high irradiance → amplify PV features in z_net
            gate = torch.sigmoid(self.weather_gate(z_w))  # [N, emb_dim]
            z_net = z_net * gate

            z = torch.cat([z_net, z_w, z_time], dim=-1)
        else:
            z = torch.cat([z_net, z_time], dim=-1)

        batch.x = self.integration(z)  # [N, emb_dim]
        return batch
