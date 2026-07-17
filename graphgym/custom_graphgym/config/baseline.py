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

    # SVR baseline (kernel quantile regression via Random Fourier Features --
    # see baseline_svr.py for why this replaces exact dual-form SVR)
    cfg.baseline.svr_rff_dim = 256
    cfg.baseline.svr_gamma = 1.0
    cfg.baseline.svr_seed = 0

    # KNN baseline (per-node analog-ensemble lookup, see baseline_knn.py)
    cfg.baseline.knn_k = 20
    cfg.baseline.knn_feature_mode = "window_calendar"  # 'window' | 'window_calendar' | 'window_calendar_weather'
    cfg.baseline.knn_distance = "euclidean"
    # The neighbor bank is dense per node ([num_nodes, S_train, F]) and can
    # be large (num_nodes x S_train x F x 4 bytes) -- defaults to CPU until
    # profiled against the real dataset size (see plan's verification
    # section); set to "cuda" once confirmed to fit in GPU memory.
    cfg.baseline.knn_cache_device = "cpu"
