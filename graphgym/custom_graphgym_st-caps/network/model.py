import torch
import torch.nn as nn
from typing import Dict
from ..encoder.sgcapl import SGCAPL
from ..head.generative import GenerativeDecoder
from ..layer.dict_learn import SparseCodingModule
from ..head.capsule import CapsuleRegressor
from ..loader.graphs import GraphBuilder
from ..loss.losses import STSGCCapsLoss

from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register

@register_network('st_sgc_caps')
class STSGCCaps(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        
        # Use values from our custom config
        config = cfg.st_caps
        self.config = config
        self.m = config.m
        self.m_prime = config.m_prime
        self.graph_builder = GraphBuilder(lambda_threshold=config.lambda_graph)
        self.encoder = SGCAPL(
            input_dim=dim_in, # Using dim_in from GraphGym
            hidden_dim=config.M_prime,
            output_dim=config.M_prime,
            n_layers=config.M,
            n_heads=config.R
        )
        self.decoder = GenerativeDecoder(
            input_dim=config.M_prime,
            edge_hidden=config.decoder_e.layers,
            node_hidden=config.decoder_s.layers
        )
        self.sparse_coder = SparseCodingModule(
            feature_dim=config.M_prime,
            n_atoms=config.q,
            lambda_sc=config.lambda_SC
        )
        self.capsule_regressor = CapsuleRegressor({
            'q': config.q,
            'n_sites': cfg.earne_data.n_user, # Fallback to earne_data if needed
            'm': config.m,
            'm_prime': config.m_prime,
            'routing_iterations': config.routing_iterations
        })
        self.loss_fn = STSGCCapsLoss(
            lambda_S=config.lambda_S,
            lambda_L=config.lambda_L,
            lambda_PV=config.lambda_PV,
            lambda_SC=config.lambda_SC
        )

    def forward(self, batch):
        # GraphGym passes a batch object
        net_demand = batch.x # Assuming [BatchSize, SeqLen, 1]
        
        # If batch.x is [TotalNodes, SeqLen, 1], we might need to handle it differently
        # But STSGCCaps was designed for [BatchSize, SeqLen] (single site history)
        # and it builds a graph across the batch.
        
        # In EARNe, batch.x is [TotalNodes, SeqLen, 1] where TotalNodes = BatchSize * NodesPerGraph
        # Here we assume net_demand is [N, T, 1]
        if net_demand.dim() == 3 and net_demand.shape[-1] == 1:
            net_demand = net_demand.squeeze(-1) # [N, T]
            
        batch_size = net_demand.shape[0]
        device = net_demand.device
        graphs = []
        for t in range(self.m + 1):
            if t == 0:
                adj = torch.zeros(batch_size, batch_size, device=device)
                nodes = net_demand[:, t]
            else:
                adj, nodes = self.graph_builder.build_graph(net_demand[:, :t + 1])
                adj = adj.to(device)
            graphs.append((adj, nodes))

        latent_features = []
        h_states, c_states = None, None
        for t, (adj, nodes) in enumerate(graphs):
            if nodes.dim() == 1:
                nodes = nodes.unsqueeze(-1)
            z_t, h_states, c_states = self.encoder(nodes, adj, h_states, c_states)
            latent_features.append(z_t)

        Z = torch.stack(latent_features)
        edge_pred, node_pred = self.decoder(Z[-1])
        codes = self.sparse_coder(Z, update_dict=self.training)
        load_pred, pv_pred = self.capsule_regressor(codes, n_sites=batch_size)

        # Attach outputs to batch for custom loss
        batch.st_caps_outputs = {
            'load_pred':       load_pred,
            'pv_pred':         pv_pred,
            'edge_pred':       edge_pred,
            'edge_true':       graphs[-1][0],
            'node_pred':       node_pred.squeeze(-1),
            'node_true':       graphs[-1][1],
            'features':        Z[-1],
            'codes':           codes[-1],
            'latent_features': Z
        }
        if hasattr(self.sparse_coder.dict_learner, 'dictionary'):
            sc_reconstruction = torch.matmul(codes[-1], self.sparse_coder.dict_learner.dictionary.T)
            batch.st_caps_outputs['sc_reconstruction'] = sc_reconstruction
        
        # Register batch for loss function access
        register.batch = batch
        
        # GraphGym expects (pred, true)
        # We concatenate load and pv predictions for the main loss if needed
        # or just return them as a stack.
        pred = torch.stack([load_pred, pv_pred], dim=1) # [N, 2]
        
        # Ground truths from batch
        true = torch.stack([batch.y_load, batch.y_pv, batch.mask], dim=1) # [N, 3]
        
        return pred, true
