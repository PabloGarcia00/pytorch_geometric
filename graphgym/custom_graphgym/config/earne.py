import os

from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config

@register_config('earne')
def set_cfg_earne(cfg):
    r"""This function sets the default config value for EARNe customized options
    :return: customized configuration used by the experiment.
    """

    # ----------------------------------------------------------------------- #
    # Data options
    # ----------------------------------------------------------------------- #
    # We use a sub-node for EARNe-specific data to avoid namespace collisions
    cfg.earne_data = CN() 
    cfg.earne_data.cadence_minutes = 15
    cfg.earne_data.days = 365
    cfg.earne_data.n_user = 10
    cfg.earne_data.base_load = 300
    cfg.earne_data.peak_load = 1500
    cfg.earne_data.raw_data_path = "../../exploratory-data-analysis/data/"
    cfg.earne_data.processed_root = "datasets/earne"
    cfg.earne_data.include_id_path = "three_macs.csv"

    # Hive & Path Configs (Relative to graphgym root, or override with EARNE_DATA_ROOT env var)
    eda_root = os.environ.get('EARNE_DATA_ROOT', '../../exploratory-data-analysis')
    cfg.earne_data.clean_hive = f"{eda_root}/output/clean_data_hive"
    cfg.earne_data.weather_hive = f"{eda_root}/output/weather_hive"
    cfg.earne_data.mapping_csv = f"{eda_root}/output/weather_hive/zip_to_station_mapping.csv"
    cfg.earne_data.metadata_csv = f"{eda_root}/output/mac_overview_updated.csv"
    cfg.earne_data.zipcode_coords = f"{eda_root}/assets/zipcode_coordinate.csv"

    # Experiment Configs
    cfg.earne_data.k_neighbors = 5
    cfg.earne_data.coastal_zips = [14, 15, 16, 17, 18, 19]
    cfg.earne_data.arnhem_zips = [68, 69]
    
    # Advanced Filtering for Transfer/Ablation Experiments
    cfg.earne_data.filter_zips = []      # e.g., [68, 69] to isolate Arnhem
    cfg.earne_data.filter_macs = []      # e.g., ['MAC1', 'MAC2'] for specific transfer sets
    cfg.earne_data.mask_weather = False  # Information Replacement Test toggle
    
    cfg.earne_data.weather_features = [
        "solar_radiation_avg", "solar_radiation_max", 
        "sunshine_duration_min", "air_temperature", "soil_temp_5cm"
    ]
    cfg.earne_data.temporal_features = ["month", "weekday", "hour"]
    cfg.earne_data.graph_mode = 'spatial_knn' # 'spatial_knn', 'full_graph', or 'learned_corr'
    cfg.earne_data.norm_mode = 'minmax' # 'minmax' or 'zero_log'
    cfg.earne_data.dual_read = False

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.model' node. We append new keys to it.
    cfg.model.node_encoder_name = 'earne_temporal'
    cfg.model.edge_encoder_name = 'none'
    cfg.model.head_name = 'earne_quantile'
    cfg.model.seq_len = 96
    cfg.model.dim_in = 1
    cfg.conv1d_kernel_size = 48

    cfg.model.hidden_channels = 64
    cfg.model.n_quantiles = 3
    cfg.model.quantiles = [0.1, 0.5, 0.9]

    # ----------------------------------------------------------------------- #
    # Train options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.train' node.
    cfg.train.physics_weight = 0.0
    cfg.train.train_split = 0.7
    cfg.train.val_split = 0.85

    # Standard GraphGym keys (overwriting defaults if necessary)
    cfg.train.mode = 'standard'
    cfg.train.epochs = 100
    cfg.train.batch_size = 32
    # Note: GraphGym typically uses cfg.optim.base_lr rather than cfg.train.lr
    cfg.optim.base_lr = 0.001

    # Custom metrics list
    cfg.custom_metrics = []
    cfg.metric_best = 'loss'
    cfg.metric_agg = 'argmin'
