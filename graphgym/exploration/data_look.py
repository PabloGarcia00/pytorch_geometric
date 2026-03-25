import torch
import pandas as pd
import numpy as np
import joblib
import duckdb
from pathlib import Path
from torch_geometric.graphgym.config import cfg, set_cfg
from custom_graphgym.config.earne import set_cfg_earne
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset

# ==========================================
# 1. SETUP CONFIGURATION
# ==========================================
set_cfg(cfg)
set_cfg_earne(cfg)

PROCESSED_ROOT = "datasets/earne"
cfg.earne_data.processed_root = PROCESSED_ROOT
cfg.earne_data.days = 365 

# ==========================================
# 2. VALIDATION: DYNAMIC TOPOLOGY (k=0, k=5, full)
# ==========================================
print("\n--- Testing Spatial Topology Modes ---")
for mode, k in [('spatial_knn', 0), ('spatial_knn', 5), ('full_graph', None)]:
    cfg.earne_data.graph_mode = mode
    cfg.earne_data.k_neighbors = k if k is not None else 0
    ds = EARNeGraphDataset(root=PROCESSED_ROOT)
    sample = ds[0]
    print(f"Mode: {mode}, k: {k} -> Edges: {sample.edge_index.shape[1]}")

# ==========================================
# 3. VALIDATION: REGIONAL FILTERING (Arnhem vs Coastal)
# ==========================================
print("\n--- Testing Regional Filtering (Transfer Experiment) ---")
# Reset filters
cfg.earne_data.filter_zips = [68, 69] # Arnhem
ds_arnhem = EARNeGraphDataset(root=PROCESSED_ROOT)
print(f"Arnhem (Zips 68, 69) Nodes: {ds_arnhem.num_nodes}")
print(f"Sample Macs: {ds_arnhem.active_macs[:3]}")

cfg.earne_data.filter_zips = [14, 15, 16, 17, 18, 19] # Coastal
ds_coastal = EARNeGraphDataset(root=PROCESSED_ROOT)
print(f"Coastal Nodes: {ds_coastal.num_nodes}")

# ==========================================
# 4. VALIDATION: WEATHER MASKING (Information Replacement)
# ==========================================
print("\n--- Testing Weather Masking ---")
cfg.earne_data.filter_zips = [] # Reset filter
cfg.earne_data.weather_features = ["solar_radiation_avg", "air_temperature"]

# No Masking
cfg.earne_data.mask_weather = False
ds_full = EARNeGraphDataset(root=PROCESSED_ROOT)
w_full = ds_full[0].weather
print(f"Weather (No Mask) Mean: {w_full[:, :2].mean():.4f} (expected non-zero)")

# With Masking
cfg.earne_data.mask_weather = True
ds_masked = EARNeGraphDataset(root=PROCESSED_ROOT)
w_masked = ds_masked[0].weather
print(f"Weather (Masked) Mean: {w_masked[:, :2].mean():.4f} (expected 0.0)")
print(f"Temporal Features (Preserved) Mean: {w_masked[:, 2:].mean():.4f} (expected non-zero)")

# ==========================================
# 5. DATA ALIGNMENT CHECK
# ==========================================
print("\n--- Testing Node Alignment ---")
ds = EARNeGraphDataset(root=PROCESSED_ROOT)
sample = ds[100]
# Check if MAC count matches Node count in tensors
assert len(sample.macs) == sample.num_nodes
assert sample.x.shape[0] == sample.num_nodes
assert sample.weather.shape[0] == sample.num_nodes
print(f"Alignment Verified for {sample.num_nodes} nodes.")

# ==========================================
# 6. SCALER & TARGET PACKAGING
# ==========================================
print("\n--- Testing Target Packaging ---")
print(f"Target Packet Shape (y): {sample.y.shape} (Load, PV, Mask)")
y_load, y_pv, y_mask = sample.y[:, 0], sample.y[:, 1], sample.y[:, 2]
print(f"Target Load Mean: {y_load.mean():.4f}")
