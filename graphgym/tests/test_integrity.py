import os
import sys
import torch
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'graphgym'))

from torch_geometric.graphgym.config import cfg, load_cfg
import custom_graphgym  # noqa

def test_integrity():
    print("--- [1/5] Checking Dataset Bundle ---")
    bundle_path = 'datasets/earne/processed/earne_master_bundle.pt'
    if not os.path.exists(bundle_path):
        print(f"FAILED: Bundle not found at {bundle_path}")
        return
    
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    
    # Check dimensions
    T, N = bundle['load_scaled'].shape
    print(f"Time steps: {T}, Nodes: {N}")
    
    # Check for NaN
    for key in ['load_scaled', 'pv_scaled', 'net_scaled']:
        if torch.isnan(bundle[key]).any():
            print(f"FAILED: NaNs detected in {key}")
        else:
            print(f"OK: No NaNs in {key}")

    print("\n--- [2/5] Checking Scaler & Physics ---")
    scaler_path = 'datasets/earne/scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"FAILED: Scaler not found at {scaler_path}")
        return
    
    scaler = joblib.load(scaler_path)
    
    # Take a sample slice
    idx = 1000
    s_load = bundle['load_scaled'][idx, 0].item()
    s_pv = bundle['pv_scaled'][idx, 0].item()
    s_net = bundle['net_scaled'][idx, 0].item()
    
    # Denormalize
    u_load = scaler.inverse_transform([[s_load]])[0,0]
    u_pv = scaler.inverse_transform([[s_pv]])[0,0]
    u_net = scaler.inverse_transform([[s_net]])[0,0]
    
    # Physics check: Net = Load - PV (with small tolerance for float/scaling)
    expected_net = u_load - u_pv
    diff = abs(u_net - expected_net)
    print(f"Denorm Load: {u_load:.2f} W")
    print(f"Denorm PV:   {u_pv:.2f} W")
    print(f"Denorm Net:  {u_net:.2f} W")
    print(f"Expected:    {expected_net:.2f} W")
    
    if diff < 1.0: # 1 Watt tolerance
        print(f"OK: Physics consistency (Net = Load - PV) verified (diff={diff:.4f})")
    else:
        print(f"WARNING: Large physics discrepancy detected: {diff:.4f}")

    print("\n--- [3/5] Checking Temporal Encoding ---")
    # Temporal: [month_s, month_c, day_s, day_c, hour_s, hour_c]
    temp = bundle['temporal_data']
    # Check if within [-1, 1]
    if (temp.min() >= -1.01) and (temp.max() <= 1.01):
        print("OK: Temporal features are normalized [-1, 1]")
    else:
        print(f"FAILED: Temporal features out of range: {temp.min()}, {temp.max()}")

    # Check if sin^2 + cos^2 approx 1
    m_check = (temp[:, 0]**2 + temp[:, 1]**2).mean().item()
    h_check = (temp[:, 4]**2 + temp[:, 5]**2).mean().item()
    print(f"Month cyclicity: {m_check:.4f}")
    print(f"Hour cyclicity:  {h_check:.4f}")
    if abs(m_check - 1.0) < 0.05 and abs(h_check - 1.0) < 0.05:
        print("OK: Sine/Cosine encoding is valid")
    else:
        print("FAILED: Sine/Cosine encoding is invalid")

    print("\n--- [4/5] Checking Metadata Alignment ---")
    if len(bundle['macs']) == N and len(bundle['zips']) == N and len(bundle['pos']) == N:
        print(f"OK: MACs, Zips, and Positions match Node count ({N})")
    else:
        print(f"FAILED: Metadata mismatch. Macs:{len(bundle['macs'])}, Nodes:{N}")

    print("\n--- [5/5] Checking Model Forward Pass ---")
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.loader import create_loader
    
    # Load a sample config to init GraphGym structures
    cfg_path = 'configs/pyg/earne_weather_all_15min.yaml'
    if os.path.exists(cfg_path):
        load_cfg(cfg, type('Args', (), {'cfg_file': cfg_path, 'opts': []}))
        cfg.accelerator = 'cpu'
        
        try:
            model = create_model()
            loaders = create_loader()
            test_loader = loaders[2]
            batch = next(iter(test_loader))
            pred, _ = model(batch)
            print(f"OK: Model forward pass successful. Output shape: {pred.shape}")
        except Exception as e:
            print(f"FAILED: Model test failed with error: {e}")
    else:
        print("SKIPPED: Could not find config to test model pass.")

if __name__ == "__main__":
    test_integrity()
