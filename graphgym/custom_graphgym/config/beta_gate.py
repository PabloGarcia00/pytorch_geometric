from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('beta_gate')
def set_cfg_beta_gate(cfg):
    r"""Hyperparameters for EARNeBetaGateHead / earne_beta_gate_loss -- the
    CVAE baseline's Gaussian-load + zero-inflated-Beta-PV technique
    (see custom_graphgym/network/baseline_cvae_network.py), ported onto
    earne_network.py's ST-GNN/LSTM/MLP shell as a plain cfg.model.head_name
    + cfg.model.loss_fun swap. Selected only when both are set to
    "earne_beta_gate"/"earne_beta_gate_loss" -- the default head_name
    ("earne_quantile") and loss_fun ("earne_loss") are untouched, so
    existing configs keep training/behaving exactly as before.
    """
    cfg.beta_gate = CN()
    cfg.beta_gate.solar_weight = 1.0
    cfg.beta_gate.gate_weight = 1.0
    cfg.beta_gate.daytime_threshold = 0.005
    cfg.beta_gate.pv_eps = 0.001
