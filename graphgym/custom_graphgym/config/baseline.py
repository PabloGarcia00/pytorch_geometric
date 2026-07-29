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

    # Linear baseline (per-node linear quantile regression, see baseline_linear.py)
    # 'window' = flattened seq_len-step history (the original default);
    # 'current' = only the most recent timestep's raw signal + calendar
    # anchor -- see results_archives/ for the write-up motivating this.
    cfg.baseline.linear_feature_mode = "window"  # 'window' | 'current'

    # SVR baseline (kernel quantile regression via Random Fourier Features --
    # see baseline_svr.py for why this replaces exact dual-form SVR)
    cfg.baseline.svr_rff_dim = 256
    # Mirrors sklearn's gamma='scale' default (1 / (n_features * X.var())),
    # fit from the first training batch since RFF needs gamma before any
    # data pass -- see baseline_svr.py. svr_gamma is only used as a fixed
    # override when this is False; a flat gamma=1.0 badly mismatches this
    # model's ~300-600 dim flattened-window input (see the investigation
    # that added this: RFF phases blew up to std~24, degenerating the model
    # to a per-node bias-only fit that ignored the actual input entirely).
    cfg.baseline.svr_gamma_auto = True
    cfg.baseline.svr_gamma = 1.0
    cfg.baseline.svr_seed = 0
    cfg.baseline.svr_feature_mode = "window"  # 'window' | 'current'

    # KNN baseline (per-node analog-ensemble lookup, see baseline_knn.py)
    cfg.baseline.knn_k = 20
    cfg.baseline.knn_feature_mode = "window_calendar"  # 'window' | 'window_calendar' | 'window_calendar_weather' | 'current' | 'current_calendar' | 'current_calendar_weather'
    cfg.baseline.knn_distance = "euclidean"
    # The neighbor bank is dense per node ([num_nodes, S_train, F]) and can
    # be large (num_nodes x S_train x F x 4 bytes) -- profiled against the
    # real dual_csi dataset (112 households, ~73.6k train samples): 9.69GB
    # at window-mode width (F=294), 0.30GB at current-mode width (F=9),
    # both comfortably fit a single GPU, and forward()'s torch.cdist search
    # was CPU-bound (the prior default) regardless of the rest of the model
    # running on GPU -- moving it here is the single biggest lever on KNN's
    # slow eval. Override back to "cpu" only if a future population is
    # large enough that the bank no longer fits in GPU memory.
    cfg.baseline.knn_cache_device = "cuda"
