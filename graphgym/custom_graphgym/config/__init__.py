from torch_geometric.graphgym.config import cfg
from .earne import set_cfg_earne
from .st_caps import set_cfg_st_caps

set_cfg_earne(cfg)
set_cfg_st_caps(cfg)
