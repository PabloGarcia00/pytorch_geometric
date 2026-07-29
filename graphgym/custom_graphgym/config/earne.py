import os

from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config("earne")
def set_cfg_earne(cfg):
    r"""This function sets the default config value for EARNe customized options
    :return: customized configuration used by the experiment.
    """

    # ----------------------------------------------------------------------- #
    # Data options
    # ----------------------------------------------------------------------- #
    # We use a sub-node for EARNe-specific data to avoid namespace collisions
    cfg.earne_data = CN()
    cfg.earne_data.processed_root = "datasets/earne"

    # Hive & Path Configs (Relative to project root, or override with EARNE_DATA_ROOT env var)
    # Default assumes exploratory-data-analysis is a sibling directory to pytorch_geometric-single-read
    config_dir = os.path.dirname(__file__)
    # Go up from graphgym/custom_graphgym/config to reaching messm/ (parent of both projects)
    eda_root_default = os.path.abspath(
        os.path.join(
            config_dir, "..", "..", "..", "..", "exploratory-data-analysis"
        )
    )
    eda_root = os.environ.get("EARNE_DATA_ROOT", eda_root_default)

    # gold_data lives directly at the pytorch_geometric project root on this
    # machine, not under exploratory-data-analysis/output/gold_layer.
    project_root_default = os.path.abspath(
        os.path.join(config_dir, "..", "..", "..")
    )
    cfg.earne_data.gold_data = os.path.join(
        project_root_default, "fleet_gold_layer.parquet"
    )
    cfg.earne_data.zipcode_coords = f"{eda_root}/assets/zipcode_coordinate.csv"

    # Experiment Configs
    cfg.earne_data.k_neighbors = 5

    # Advanced Filtering for Transfer/Ablation Experiments
    cfg.earne_data.filter_zips = []  # e.g., [68, 69] to isolate Arnhem
    cfg.earne_data.filter_ids = (
        []
    )  # e.g., ['MAC1', 'MAC2'] for specific transfer sets
    cfg.earne_data.max_nodes = 0  # 0 = no limit; N > 0 = keep first N nodes
    cfg.earne_data.mask_physics_impossible = False  # mask timesteps where generation_w > inverter_w
    cfg.earne_data.require_full_span = True  # True = drop nodes absent in first/last week (optimal coverage); False = native observation window
    cfg.earne_data.start_date = "2023-04-03"
    cfg.earne_data.end_date = "2026-04-03"
    cfg.earne_data.weather_mode = False  # Information Replacement Test toggle
    cfg.earne_data.dual_read = False

    cfg.earne_data.weather_features = []

    cfg.earne_data.graph_mode = "spatial_knn"  # 'spatial_knn', 'full_graph', or 'learned_corr' or 'static_corr'
    cfg.earne_data.dual_agg = "concat"
    cfg.earne_data.energy_norm_mode = CN()
    cfg.earne_data.energy_norm_mode.load = "log1p"
    cfg.earne_data.energy_norm_mode.pv = "ihs"
    cfg.earne_data.energy_norm_mode.net_demand = "ihs"
    cfg.earne_data.energy_norm_mode.generation = "ihs"
    cfg.earne_data.energy_norm_mode.consumption = "log1p"

    cfg.earne_data.weather_norm_mode = CN()
    cfg.earne_data.weather_norm_mode.all_features = "znorm"

    cfg.earne_data.lambda_graph = (
        0.25  # threshold for learned_corr (if not using topk)
    )
    cfg.earne_data.use_topk = True  # force fixed edge count per node

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.model' node. We append new keys to it.
    cfg.model.node_encoder_name = "earne_temporal"
    cfg.model.edge_encoder_name = "none"
    cfg.model.head_name = "earne_quantile"
    cfg.model.seq_len = 96
    cfg.model.dim_in = 1
    cfg.conv1d_kernel_size = 47  # odd kernel avoids redundant zero-padding copy in Conv1d same-padding
    cfg.conv1d_weather_kernel_size = (
        7  # odd; separate kernel for weather stream
    )

    cfg.model.hidden_channels = 64
    cfg.model.n_quantiles = 3
    cfg.model.quantiles = [0.1, 0.5, 0.9]
    # Which component(s) to predict: subset of ["load", "pv"]. Applies to
    # every registered network (earne_network, baseline_linear/svr/knn,
    # baseline_cvae, st_sgc_caps) via custom_graphgym/target_utils.py's
    # active_targets(). Output column order is always load-before-pv when
    # both are selected, regardless of the order listed here.
    cfg.model.predict_targets = ["pv"]

    # ----------------------------------------------------------------------- #
    # Train options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.train' node.
    cfg.train.physics_weight = 0.0
    cfg.train.crossing_weight = 0.05
    cfg.train.train_split = 0.6
    cfg.train.val_split = 0.8

    # Standard GraphGym keys (overwriting defaults if necessary)
    cfg.train.mode = "earne_train"
    cfg.train.epochs = 100
    cfg.train.batch_size = 32

    # Note: GraphGym typically uses cfg.optim.base_lr rather than cfg.train.lr
    cfg.optim.base_lr = 0.001
    cfg.optim.scheduler = "warmup_cos"

    # Early stopping & checkpointing
    cfg.train.limit_batches = 0  # 0 = no limit; N > 0 = cap train/val/test steps per epoch

    cfg.train.early_stopping = CN()
    cfg.train.early_stopping.enable = False
    cfg.train.early_stopping.monitor = "val_loss"
    cfg.train.early_stopping.patience = 10
    cfg.train.early_stopping.mode = "min"
    cfg.train.early_stopping.save_last = True
    # Lightning's EarlyStopping defaults min_delta to 0.0 -- ANY improvement,
    # even sub-noise-level floating-point fluctuation, resets the patience
    # counter. Harmless for models that descend noisily over many epochs,
    # but a model that converges quickly (e.g. baseline_linear/svr in
    # 'current' feature_mode, far fewer parameters to fit) can plateau
    # almost immediately and then oscillate within this threshold forever,
    # never actually early-stopping. 0.0 preserves prior behavior exactly;
    # set per-config where a genuine plateau needs to trigger a stop.
    cfg.train.early_stopping.min_delta = 0.0

    # Warmup
    cfg.train.warmup = CN()
    cfg.train.warmup.enable = False
    cfg.train.warmup.epochs = 5

    # loggers
    cfg.train.wandb = CN()
    cfg.train.wandb.project = "earne"
    cfg.train.wandb.run_name = ""

    # Custom metrics list
    cfg.custom_metrics = []
    cfg.metric_best = "loss"
    cfg.metric_agg = "argmin"
