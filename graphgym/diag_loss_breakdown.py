import sys
import torch

import custom_graphgym  # noqa, register custom modules
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.cmd_args import parse_args
from custom_graphgym.loss.earne_loss import masked_quantile_loss

cfg_file = sys.argv[1]
ckpt_path = sys.argv[2]

sys.argv = ["diag.py", "--cfg", cfg_file]
args = parse_args()
load_cfg(cfg, args)
# NOTE: deliberately not calling set_out_dir/set_run_dir here - those call
# makedirs_rm_exist() and would wipe the real training results for this cfg.
cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
cfg.devices = 1

datamodule = GraphGymDataModule()
model = create_model()

state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

val_loader = datamodule.val_dataloader()
quantiles = cfg.model.quantiles
n_q = len(quantiles)
q50_idx = quantiles.index(0.5)

totals = {"loss_load": 0.0, "loss_pv": 0.0, "loss_physics_raw": 0.0, "n_batches": 0}

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)
        pred, true = model(batch)

        q_load = pred[:, :n_q]
        q_pv = pred[:, n_q:]
        y_load = true[:, 0]
        y_pv = true[:, 1]
        mask = true[:, 2]
        y_net_demand = true[:, 3]

        loss_load = masked_quantile_loss(q_load, y_load, quantiles, mask)
        loss_pv = masked_quantile_loss(q_pv, y_pv, quantiles, mask)

        q_pv_phys = q_pv[:, q50_idx]
        pred_net_demand = q_load[:, q50_idx] - q_pv_phys
        diff_sq = (pred_net_demand - y_net_demand) ** 2
        loss_physics_raw = (diff_sq * mask).sum() / (mask.sum() + 1e-9)

        totals["loss_load"] += loss_load.item()
        totals["loss_pv"] += loss_pv.item()
        totals["loss_physics_raw"] += loss_physics_raw.item()
        totals["n_batches"] += 1

        if i >= 49:  # cap at 50 batches for speed
            break

n = totals["n_batches"]
print(f"cfg={cfg_file}")
print(f"ckpt={ckpt_path}")
print(f"n_batches_sampled={n}")
print(f"mean loss_load          = {totals['loss_load']/n:.6f}")
print(f"mean loss_pv            = {totals['loss_pv']/n:.6f}")
print(f"mean loss_physics (raw) = {totals['loss_physics_raw']/n:.6f}")
print(f"physics_weight in cfg   = {cfg.train.physics_weight}")
print(f"weighted physics contrib= {cfg.train.physics_weight * totals['loss_physics_raw']/n:.6f}")
print(f"sum(load+pv)            = {(totals['loss_load']+totals['loss_pv'])/n:.6f}")
