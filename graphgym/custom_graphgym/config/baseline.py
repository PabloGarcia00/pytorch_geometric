from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('baseline')
def set_cfg_baseline(cfg):
    r"""Hyperparameters for the non-GNN baseline models (MLP, LSTM, CVAE)
    that plug into EARNeNetwork's node_encoder slot or run as a standalone
    @register_network (CVAE). Values carried from the tuned baselines/configs/
    *.yaml overrides, not the standalone classes' constructor defaults.
    """
    cfg.baseline = CN()

    # MLP encoder
    cfg.baseline.mlp_hidden_dim = 128

    # LSTM (BiLSTM) encoder
    cfg.baseline.lstm_hidden_dim = 32
    cfg.baseline.lstm_layers = 1
    cfg.baseline.lstm_dropout = 0.1

    # CVAE network
    cfg.baseline.cvae_latent_dim = 32
    cfg.baseline.cvae_hidden_dim = 64
    cfg.baseline.cvae_decoder_dim = 64
    cfg.baseline.cvae_num_layers = 2
    cfg.baseline.cvae_dropout = 0.1
    cfg.baseline.cvae_solar_weight = 1.0
    cfg.baseline.cvae_gate_weight = 1.0
    cfg.baseline.cvae_kl_weight = 0.1
    cfg.baseline.cvae_daytime_threshold = 0.005
    cfg.baseline.cvae_pv_eps = 0.001
