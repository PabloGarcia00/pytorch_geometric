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
    cfg.earne_data.raw_data_path = "datasets/earne/raw/"
    cfg.earne_data.processed_root = "./datasets/earne/"
    cfg.earne_data.include_id_path = "three_macs.csv"

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.model' node. We append new keys to it.
    cfg.model.node_encoder_name = 'earne_temporal'
    cfg.model.edge_encoder_name = 'none'
    cfg.model.head_name = 'earne_quantile'
    cfg.model.seq_len = 96
    cfg.conv1d_kernel_size = 48
    cfg.model.hidden_channels = 64
    cfg.model.n_quantiles = 3
    cfg.model.quantiles = [0.1, 0.5, 0.9]

    # ----------------------------------------------------------------------- #
    # Train options
    # ----------------------------------------------------------------------- #
    # GraphGym already has a 'cfg.train' node.
    cfg.train.physics_weight = 0.1
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
