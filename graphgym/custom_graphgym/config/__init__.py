from torch_geometric.graphgym.config import cfg
from .earne import set_cfg_earne
from .example import set_cfg_example

set_cfg_earne(cfg)
set_cfg_example(cfg)
