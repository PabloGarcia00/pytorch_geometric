"""One-time script to generate the EARNe processed bundle (data.pt + transform_single.pt).

Output lands in baselines/earne_bundle/processed/data.pt
                  baselines/earne_bundle/transform_single.pt

Run once; both train.py and evaluate.py will find it via data_root: earne_bundle.
"""

import sys
from pathlib import Path

# ── graphgym on sys.path ──────────────────────────────────────────────────────
graphgym_root = Path(__file__).resolve().parent.parent / "graphgym"
sys.path.insert(0, str(graphgym_root))

import custom_graphgym  # noqa — registers all custom configs + loaders
from torch_geometric.graphgym.config import cfg

# ── Minimal config ────────────────────────────────────────────────────────────
cfg.model.dim_in   = 1
cfg.model.seq_len  = 96
cfg.train.train_split = 0.6
cfg.train.val_split   = 0.8

cfg.earne_data.start_date  = "2023-04-03"
cfg.earne_data.require_full_span = True
cfg.earne_data.filter_zips = []
cfg.earne_data.filter_ids  = []
cfg.earne_data.max_nodes   = 0
cfg.earne_data.mask_physics_impossible = False
cfg.earne_data.graph_mode  = "spatial_knn"
cfg.earne_data.k_neighbors = 5
cfg.earne_data.use_weather = False
cfg.earne_data.weather_features = []

cfg.earne_data.energy_norm_mode.load       = "log1p"
cfg.earne_data.energy_norm_mode.pv         = "ihs"
cfg.earne_data.energy_norm_mode.net_demand = "ihs"

# ── Output path ───────────────────────────────────────────────────────────────
out_root = Path(__file__).resolve().parent / "earne_bundle"
out_root.mkdir(parents=True, exist_ok=True)
(out_root / "processed").mkdir(exist_ok=True)

print(f"Bundle will be written to: {out_root}")

# ── Trigger processing ────────────────────────────────────────────────────────
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset

dataset = EARNeGraphDataset(root=str(out_root), seq_len=cfg.model.seq_len)

print(f"\nDone. Nodes: {dataset.num_nodes}  Samples: {dataset.len()}")
print(f"  data.pt          → {out_root / 'processed' / 'data.pt'}")
print(f"  transform_single → {out_root / 'transform_single.pt'}")
