import sys
import torch
import torch.nn.functional as F

import custom_graphgym  # noqa, register custom modules
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.cmd_args import parse_args

cfg_file = sys.argv[1]
ckpt_path = sys.argv[2]
max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 500

sys.argv = ["diag.py", "--cfg", cfg_file]
args = parse_args()
load_cfg(cfg, args)
# NOTE: deliberately not calling set_out_dir/set_run_dir - would wipe real results.
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

worst = []  # (total, idx, components, mask_sum)

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)
        pred, true = model(batch)
        outputs = batch.st_caps_outputs

        components = {}

        j_E = F.mse_loss(outputs['edge_pred'], outputs['edge_true'])
        j_S = F.mse_loss(outputs['node_pred'], outputs['node_true'])
        recon_loss = j_E + cfg.st_caps.lambda_S * j_S
        components['edge_loss'] = j_E.item()
        components['node_loss'] = j_S.item()
        components['recon_loss'] = recon_loss.item()

        sc_loss_val = None
        if 'features' in outputs and 'sc_reconstruction' in outputs:
            recon_error = F.mse_loss(outputs['sc_reconstruction'], outputs['features'])
            l1_penalty = torch.mean(torch.abs(outputs['codes']))
            sc_loss_val = recon_error + cfg.st_caps.lambda_SC * l1_penalty
            components['sc_recon_error'] = recon_error.item()
            components['sc_l1_penalty'] = l1_penalty.item()
            components['sc_loss'] = sc_loss_val.item()

        j_L = F.mse_loss(outputs['load_pred'], outputs['load_true'])
        j_PV = F.mse_loss(outputs['pv_pred'], outputs['pv_true'])
        disagg_loss = cfg.st_caps.lambda_L * j_L + cfg.st_caps.lambda_PV * j_PV
        components['load_loss'] = j_L.item()
        components['pv_loss'] = j_PV.item()
        components['disagg_loss'] = disagg_loss.item()

        mask = true[:, 2]
        y_net_demand = true[:, 3]
        pred_net_demand = outputs['load_pred'] - outputs['pv_pred']
        diff_sq = (pred_net_demand - y_net_demand) ** 2
        physics_loss_raw = (diff_sq * mask).sum() / (mask.sum() + 1e-9)
        components['physics_loss_raw'] = physics_loss_raw.item()
        components['mask_sum'] = mask.sum().item()
        components['mask_numel'] = mask.numel()

        total = recon_loss.item() + (sc_loss_val.item() if sc_loss_val is not None else 0.0) + disagg_loss.item()
        components['total_without_physics'] = total

        worst.append((total, i, dict(components)))

        if i >= max_batches:
            break

worst.sort(key=lambda x: -x[0])
print(f"Sampled {len(worst)} val batches from {cfg_file}")
print("Top 10 worst (highest total_without_physics) batches:")
for total, idx, comp in worst[:10]:
    print(f"  batch_idx={idx}  total={total:.3f}  " + "  ".join(f"{k}={v:.4f}" for k, v in comp.items()))

print()
print("Bottom 5 (typical) batches:")
for total, idx, comp in worst[-5:]:
    print(f"  batch_idx={idx}  total={total:.3f}  " + "  ".join(f"{k}={v:.4f}" for k, v in comp.items()))
