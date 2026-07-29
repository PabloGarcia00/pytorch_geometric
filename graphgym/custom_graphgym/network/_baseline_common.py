"""Shared building blocks for the per-node classical baselines (linear
regression, SVR, KNN). All three bypass earne_network's encoder/GNN-body/head
chain entirely (they register as standalone @register_network modules) so
that per-node fitting is structurally guaranteed rather than merely
configured -- see baseline_cvae_network.py / st_sgc_caps.py for the same
bypass pattern used elsewhere in this project.
"""

import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg


def node_ids_for_batch(batch, num_nodes: int) -> torch.Tensor:
    """Fixed node-within-graph id for every node in a disjoint-unioned Batch.

    Every EARNeGraphDataset sample is a full snapshot of the same
    `num_nodes` households in the same fixed order (EARNeGraphDataset.get),
    and PyG's disjoint union concatenates each graph's node block
    contiguously -- so node identity within the flattened batch is simply
    `arange(num_nodes)` tiled once per graph, no lookup required.
    """
    B = batch.num_graphs
    return torch.arange(num_nodes, device=batch.x.device).repeat(B)


def flatten_window(batch, weather_mode: bool) -> torch.Tensor:
    """[N, T, C] -> [N, T*C]. Mirrors baseline_mlp_encoder.py's input
    construction (net-demand/consumption history + operational flag [+
    weather]), so all three classical baselines see the same features the
    MLP/LSTM encoder ablations do.
    """
    x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_x]
    if weather_mode:
        x_in = torch.cat([x_in, batch.weather], dim=-1)
    N = x_in.shape[0]
    return x_in.reshape(N, -1)


def last_step_calendar_features(batch, num_nodes: int) -> torch.Tensor:
    """[N, 6] cyclical (month/day-of-week/hour sin-cos) calendar encoding of
    the input window's most recent observed step, broadcast to every node.

    batch.temporal holds the *input window*'s calendar encoding
    (`temporal=self.temporal_data[idx:tgt]` in EARNeGraphDataset.get) -- the
    data contract has no separate target-step calendar field, so the last
    observed step is the closest available seasonal anchor.
    """
    B = batch.num_graphs
    T = cfg.model.seq_len
    temporal = batch.temporal.view(B, T, 6)
    last = temporal[:, -1, :]  # [B, 6]
    return last.repeat_interleave(num_nodes, dim=0)  # [B*num_nodes, 6]


def baseline_in_features(dim_in: int, weather_mode: bool, n_weather: int) -> int:
    """Flattened feature width produced by build_features() below."""
    seq_len = cfg.model.seq_len
    c = dim_in + 1 + (n_weather if weather_mode else 0)  # +1 = operational flag
    return seq_len * c + 6  # +6 = last_step_calendar_features


def build_features(batch, weather_mode: bool, num_nodes: int) -> torch.Tensor:
    """[N, F] per-node feature vector: flattened window + calendar anchor."""
    window = flatten_window(batch, weather_mode)
    calendar = last_step_calendar_features(batch, num_nodes)
    return torch.cat([window, calendar], dim=-1)


def current_step_features(batch, weather_mode: bool) -> torch.Tensor:
    """[N, T, C] -> [N, C], using only the most recent input timestep --
    the nowcasting counterpart to flatten_window's full-window flatten.
    PV output is close to instantaneous in current conditions (weather,
    time of day), so this trades away autoregressive history for a much
    narrower per-household feature width."""
    x_in = torch.cat([batch.x, batch.operational], dim=-1)  # [N, T, C_x]
    if weather_mode:
        x_in = torch.cat([x_in, batch.weather], dim=-1)
    return x_in[:, -1, :]


def baseline_current_in_features(dim_in: int, weather_mode: bool, n_weather: int) -> int:
    """Width counterpart to baseline_in_features, without the seq_len
    multiply -- current_step_features() has no window to flatten."""
    c = dim_in + 1 + (n_weather if weather_mode else 0)  # +1 = operational flag
    return c + 6  # +6 = last_step_calendar_features


def build_current_features(batch, weather_mode: bool, num_nodes: int) -> torch.Tensor:
    """[N, F] per-node feature vector: current-step signal + calendar anchor."""
    current = current_step_features(batch, weather_mode)
    calendar = last_step_calendar_features(batch, num_nodes)
    return torch.cat([current, calendar], dim=-1)


def stack_true(batch) -> torch.Tensor:
    """[N, 4] target tensor (y_load, y_pv, mask, y_net_demand) -- identical
    contract to earne_network.forward's `true`, consumed unmodified by
    earne_loss / earne_mae_load / earne_mae_pv / DisaggregationMetrics.
    """
    return torch.stack(
        [batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1
    )


class PerNodeLinearQuantileHead(nn.Module):
    """Grouped per-node linear map: one independent
    [in_features -> out_features] linear layer per node, batched into a
    single [num_nodes, in_features, out_features] parameter tensor.

    num_nodes must be known at construction time (not initialized lazily on
    first forward()) because GraphGymModule.configure_optimizers() collects
    self.model.parameters() before any forward pass runs -- a lazily
    materialized parameter would never make it into the optimizer's param
    groups. num_nodes is read from cfg.share.num_nodes, set by
    load_earne_dataset() (custom_graphgym/loader/graph_dataset.py) the same
    way upstream GraphGym sets cfg.share.dim_in/dim_out from the dataset.
    """

    def __init__(self, num_nodes: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_nodes, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))
        bound = 1.0 / (in_features ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        w = self.weight[node_ids]  # [B_nodes, in_features, out_features]
        b = self.bias[node_ids]  # [B_nodes, out_features]
        return torch.einsum("bf,bfo->bo", x, w) + b
