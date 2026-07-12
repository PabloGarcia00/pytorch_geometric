from torch_geometric.graphgym.config import cfg
from .earne import set_cfg_earne
from .st_caps import set_cfg_st_caps
from .baseline import set_cfg_baseline

set_cfg_earne(cfg)
set_cfg_st_caps(cfg)
set_cfg_baseline(cfg)
