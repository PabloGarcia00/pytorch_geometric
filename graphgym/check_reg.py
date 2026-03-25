import custom_graphgym
from torch_geometric.graphgym.config import cfg, load_cfg
import torch_geometric.graphgym.register as register

# Mock args
class Args:
    cfg_file = 'configs/pyg/earne_test_refactored.yaml'
    opts = []

args = Args()
load_cfg(cfg, args)

print(f"cfg.dataset.format: {cfg.dataset.format}")
print(f"Registered loaders: {list(register.loader_dict.keys())}")

if cfg.dataset.format in register.loader_dict:
    print(f"Loader for {cfg.dataset.format} is {register.loader_dict[cfg.dataset.format]}")
else:
    print(f"Loader {cfg.dataset.format} NOT found!")
