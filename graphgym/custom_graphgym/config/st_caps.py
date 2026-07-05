from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('st_caps')
def set_cfg_st_caps(cfg):
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
    cfg.st_caps.n_user = 40  # expected number of nodes; used to init capsule W matrix

    cfg.st_caps.decoder_e = CN()
    cfg.st_caps.decoder_e.layers = [128, 256, 1024, 512]

    cfg.st_caps.decoder_s = CN()
    cfg.st_caps.decoder_s.layers = [512, 256, 1024, 512, 256]
