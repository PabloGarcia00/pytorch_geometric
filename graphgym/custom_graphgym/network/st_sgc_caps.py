import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..encoder.sgcapl import SGCAPL
from ..head.generative import GenerativeDecoder
from ..head.capsule import CapsuleRegressor
from ..layer.dict_learn import SparseCodingModule
from ..loader.graphs import GraphBuilder
from ..loss.st_caps_loss import STSGCCapsLoss
from ..target_utils import active_targets


@register_network('st_sgc_caps')
class STSGCCaps(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        config = cfg.st_caps
        self.m = config.m
        self.m_prime = config.m_prime
        self.targets = active_targets()

        # Dual read: batch.x carries [consumption, generation] instead of net
        # demand, same convention as earne_network (cfg.model.dim_in == 2).
        self.dual_read = cfg.model.dim_in == 2
        self.signal_channels = 2 if self.dual_read else 1

        # Weather: gates the per-timestep encoder embedding, mirroring
        # earne_network's temporal encoder (high irradiance amplifies
        # PV-related features in the embedding).
        self.weather_mode = cfg.earne_data.weather_mode
        self.n_weather = len(cfg.earne_data.weather_features) if self.weather_mode else 0

        self.graph_mode = cfg.earne_data.get('graph_mode', 'spatial_knn')
        if self.graph_mode == 'learned_corr':
            self.lambda_threshold = nn.Parameter(torch.tensor(config.lambda_graph))
        else:
            self.lambda_threshold = None

        self.graph_builder = GraphBuilder(lambda_threshold=config.lambda_graph)
        self.encoder = SGCAPL(
            input_dim=self.signal_channels,
            hidden_dim=config.M_prime,
            output_dim=config.M_prime,
            n_layers=config.M,
            n_heads=config.R,
        )
        self.decoder = GenerativeDecoder(
            input_dim=config.M_prime,
            edge_hidden=config.decoder_e.layers,
            node_hidden=config.decoder_s.layers,
            output_dim=self.signal_channels,
        )
        self.sparse_coder = SparseCodingModule(
            feature_dim=config.M_prime,
            n_atoms=config.q,
            lambda_sc=config.lambda_SC,
        )
        self.capsule_regressor = CapsuleRegressor({
            'q': config.q,
            'n_sites': config.n_user,
            'm': config.m,
            'm_prime': config.m_prime,
            'routing_iterations': config.routing_iterations,
        })
        self.loss_fn = STSGCCapsLoss(
            lambda_S=config.lambda_S,
            lambda_L=config.lambda_L,
            lambda_PV=config.lambda_PV,
            lambda_SC=config.lambda_SC,
        )

        if self.weather_mode:
            gate_hidden = config.M_prime // 2
            self.weather_gate = nn.Sequential(
                nn.Linear(self.n_weather, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, config.M_prime),
            )

    def _extract_raw_x(self, batch):
        """Single read: batch.x [N, T, 1] -> (net_demand [N, T], None)
        Dual read:   batch.x [N, T, 2] -> (consumption [N, T], generation [N, T])
        """
        if batch.x.shape[-1] == 2:
            return batch.x[:, :, 0], batch.x[:, :, 1]
        return batch.x[:, :, 0], None

    def forward(self, batch):
        raw_x, second_stream = self._extract_raw_x(batch)

        # Graph correlation always works off a single [N, T] signal; in dual
        # read mode use net (consumption - generation), same convention as
        # earne_network._get_graph.
        graph_signal = raw_x if second_stream is None else raw_x - second_stream

        batch_size = raw_x.shape[0]
        device = raw_x.device
        weather = batch.weather if self.weather_mode else None  # [N, T, W]

        # Every timestep's graph only looks backward at its own trailing window,
        # independent of every other timestep - so all m+1 graphs are built in one
        # batched call instead of a Python loop issuing m+1 separate correlation
        # computations. See GraphBuilder.build_graphs_all for the equivalence.
        all_adj = self.graph_builder.build_graphs_all(
            graph_signal[:, :self.m + 1],
            lambda_threshold=self.lambda_threshold,
        ).to(device)  # [m+1, N, N]
        zero_adj = torch.zeros(batch_size, batch_size, device=device)

        if second_stream is None:
            node_feats_all = raw_x[:, :self.m + 1].unsqueeze(-1)  # [N, m+1, 1]
        else:
            node_feats_all = torch.stack(
                [raw_x[:, :self.m + 1], second_stream[:, :self.m + 1]], dim=-1
            )  # [N, m+1, 2]

        graphs = []
        for t in range(self.m + 1):
            adj = zero_adj if t == 0 else all_adj[t]
            graphs.append((adj, node_feats_all[:, t]))

        latent_features = []
        h_states, c_states = None, None
        for t, (adj, nodes) in enumerate(graphs):
            z_t, h_states, c_states = self.encoder(nodes, adj, h_states, c_states)
            if self.weather_mode:
                gate = torch.sigmoid(self.weather_gate(weather[:, t, :]))
                z_t = z_t * gate
            latent_features.append(z_t)

        Z = torch.stack(latent_features)        # [T, N, M_prime]
        edge_pred, node_pred = self.decoder(Z[-1])
        codes = self.sparse_coder(Z, update_dict=self.training)
        load_pred, pv_pred = self.capsule_regressor(codes, n_sites=batch_size)

        node_true = graphs[-1][1]
        if self.signal_channels == 1:
            node_pred = node_pred.squeeze(-1)
            node_true = node_true.squeeze(-1)

        batch.st_caps_outputs = {
            'load_pred':       load_pred,
            'pv_pred':         pv_pred,
            'load_true':       batch.y_load,
            'pv_true':         batch.y_pv,
            'edge_pred':       edge_pred,
            'edge_true':       graphs[-1][0],
            'node_pred':       node_pred,
            'node_true':       node_true,
            'features':        Z[-1],
            'codes':           codes[-1],
            'latent_features': Z,
        }
        if hasattr(self.sparse_coder.dict_learner, 'dictionary'):
            sc_reconstruction = torch.matmul(codes[-1], self.sparse_coder.dict_learner.dictionary.T)
            batch.st_caps_outputs['sc_reconstruction'] = sc_reconstruction

        register.batch = batch

        # load_pred/pv_pred are computed jointly above (shared capsule
        # routing) and both are always stashed in st_caps_outputs for the
        # loss, but the reported pred only includes the selected target(s)
        # -- cfg.model.predict_targets, default PV only.
        pred_by_target = {"load": load_pred, "pv": pv_pred}
        pred = torch.stack([pred_by_target[t] for t in self.targets], dim=1)  # [N, len(targets)]
        true = torch.stack([batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1)  # [N, 4]
        return pred, true
