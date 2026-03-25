import yaml
from typing import Dict, Any
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config

@register_config('st_caps')
def set_cfg_st_caps(cfg):
    r"""This function sets the default config value for ST-SGCCaps customized options
    :return: customized configuration used by the experiment.
    """
    cfg.st_caps = CN()
    cfg.st_caps.m = 35
    cfg.st_caps.m_prime = 45
    cfg.st_caps.lambda_graph = 0.25
    cfg.st_caps.M_prime = 128
    cfg.st_caps.M = 8
    cfg.st_caps.R = 3
    cfg.st_caps.q = 85
    cfg.st_caps.lambda_SC = 0.1
    cfg.st_caps.lambda_S = 1.0
    cfg.st_caps.lambda_L = 1.0
    cfg.st_caps.lambda_PV = 1.0
    cfg.st_caps.routing_iterations = 3
    
    cfg.st_caps.decoder_e = CN()
    cfg.st_caps.decoder_e.layers = [128, 256, 1024, 512]
    
    cfg.st_caps.decoder_s = CN()
    cfg.st_caps.decoder_s.layers = [512, 256, 1024, 512, 256]

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], path: str):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
