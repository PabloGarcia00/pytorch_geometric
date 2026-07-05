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


@register_network('st_sgc_caps')
class STSGCCaps(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        config = cfg.st_caps
        self.m = config.m
        self.m_prime = config.m_prime

        self.graph_mode = cfg.earne_data.get('graph_mode', 'spatial_knn')
        if self.graph_mode == 'learned_corr':
            self.lambda_threshold = nn.Parameter(torch.tensor(config.lambda_graph))
        else:
            self.lambda_threshold = None

        self.graph_builder = GraphBuilder(lambda_threshold=config.lambda_graph)
        self.encoder = SGCAPL(
            input_dim=1,  # receives one scalar per node per time step, not the full sequence
            hidden_dim=config.M_prime,
            output_dim=config.M_prime,
            n_layers=config.M,
            n_heads=config.R,
        )
        self.decoder = GenerativeDecoder(
            input_dim=config.M_prime,
            edge_hidden=config.decoder_e.layers,
            node_hidden=config.decoder_s.layers,
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

    def forward(self, batch):
        net_demand = batch.x
        if net_demand.dim() == 3 and net_demand.shape[-1] == 1:
            net_demand = net_demand.squeeze(-1)  # [N, T]

        batch_size = net_demand.shape[0]
        device = net_demand.device

        graphs = []
        for t in range(self.m + 1):
            if t == 0:
                adj = torch.zeros(batch_size, batch_size, device=device)
                nodes = net_demand[:, t]
            else:
                adj, nodes = self.graph_builder.build_graph(
                    net_demand[:, :t + 1],
                    lambda_threshold=self.lambda_threshold,
                )
                adj = adj.to(device)
            graphs.append((adj, nodes))

        latent_features = []
        h_states, c_states = None, None
        for adj, nodes in graphs:
            if nodes.dim() == 1:
                nodes = nodes.unsqueeze(-1)
            z_t, h_states, c_states = self.encoder(nodes, adj, h_states, c_states)
            latent_features.append(z_t)

        Z = torch.stack(latent_features)        # [T, N, M_prime]
        edge_pred, node_pred = self.decoder(Z[-1])
        codes = self.sparse_coder(Z, update_dict=self.training)
        load_pred, pv_pred = self.capsule_regressor(codes, n_sites=batch_size)

        batch.st_caps_outputs = {
            'load_pred':       load_pred,
            'pv_pred':         pv_pred,
            'load_true':       batch.y_load,
            'pv_true':         batch.y_pv,
            'edge_pred':       edge_pred,
            'edge_true':       graphs[-1][0],
            'node_pred':       node_pred.squeeze(-1),
            'node_true':       graphs[-1][1],
            'features':        Z[-1],
            'codes':           codes[-1],
            'latent_features': Z,
        }
        if hasattr(self.sparse_coder.dict_learner, 'dictionary'):
            sc_reconstruction = torch.matmul(codes[-1], self.sparse_coder.dict_learner.dictionary.T)
            batch.st_caps_outputs['sc_reconstruction'] = sc_reconstruction

        register.batch = batch

        pred = torch.stack([load_pred, pv_pred], dim=1)   # [N, 2]
        true = torch.stack([batch.y_load, batch.y_pv, batch.mask, batch.y_net_demand], dim=1)  # [N, 4]
        return pred, true
