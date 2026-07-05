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

def test_extended():
    bundle_path = 'datasets/earne/processed/earne_master_bundle.pt'
    cfg_path = 'configs/pyg/earne_weather_all_15min.yaml'

    print("\n--- [6/8] Quantile Config Consistency ---")
    if os.path.exists(cfg_path):
        load_cfg(cfg, type('Args', (), {'cfg_file': cfg_path, 'opts': []}))
        n_q = cfg.model.n_quantiles
        q_list = cfg.model.quantiles
        if len(q_list) == n_q:
            print(f"OK: n_quantiles ({n_q}) matches len(quantiles) ({len(q_list)})")
        else:
            print(f"FAILED: n_quantiles={n_q} but len(quantiles)={len(q_list)}")
        if 0.5 in q_list:
            print("OK: 0.5 (median) is present in quantiles list")
        else:
            print(f"FAILED: 0.5 not found in quantiles={q_list} — metric will crash")
    else:
        print("SKIPPED: Config not found")

    print("\n--- [7/8] Empty filter_zips Guard ---")
    if os.path.exists(bundle_path) and os.path.exists(cfg_path):
        load_cfg(cfg, type('Args', (), {'cfg_file': cfg_path, 'opts': []}))
        cfg.earne_data.filter_zips = [99999]
        try:
            from custom_graphgym.loader.graph_dataset import EARNeGraphDataset
            _ = EARNeGraphDataset(root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len)
            print("FAILED: Expected ValueError for empty node set, but none was raised")
        except ValueError as e:
            print(f"OK: ValueError raised as expected — {e}")
        except Exception as e:
            print(f"WARNING: Unexpected error type ({type(e).__name__}) — {e}")
        finally:
            cfg.earne_data.filter_zips = []
    else:
        print("SKIPPED: Bundle or config not found")

    print("\n--- [8/8] Timestamp Presence in Bundle ---")
    if os.path.exists(bundle_path):
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
        T = bundle['load_scaled'].shape[0]
        if 'timestamps' in bundle and bundle['timestamps'] is not None:
            ts_len = len(bundle['timestamps'])
            if ts_len == T:
                print(f"OK: timestamps present and length matches T ({T})")
            else:
                print(f"FAILED: timestamps length {ts_len} != T {T}")
        else:
            print("FAILED: 'timestamps' key missing or None in bundle")
    else:
        print("SKIPPED: Bundle not found")


def test_temporal_split():
    """
    Unit tests for EARNeGraphDataset.get_split_indices().

    All tests use a lightweight mock that patches only the attributes the method
    reads — no real data or DuckDB is required.
    """
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from custom_graphgym.loader.graph_dataset import EARNeGraphDataset

    print("\n--- [TS-1] 3-Year Case A: test set starts at year 3 ---")
    # 3 years × 365 days × 96 slots = 105120 time steps
    # seq_len = 96 → 105120 - 96 - 1 = 105023 samples
    seq_len = 96
    n_steps = 3 * 365 * 96        # one 15-min slot per step
    ts = pd.date_range("2021-01-01", periods=n_steps, freq="15min")

    ds = MagicMock(spec=EARNeGraphDataset)
    ds.seq_len = seq_len
    ds.timestamps = ts.tolist()
    # len() counts usable samples
    ds.len = lambda: n_steps - seq_len - 1

    with patch.object(EARNeGraphDataset, 'get_split_indices', EARNeGraphDataset.get_split_indices):
        train_idx, val_idx, test_idx = EARNeGraphDataset.get_split_indices(ds)

    # First test sample target ts = ts[seq_len]; test should start at 2023-01-01
    test_start_ts = ts[test_idx[0] + seq_len]
    assert test_start_ts.year == 2023, (
        f"FAILED: Expected test to start in 2023, got {test_start_ts.year}"
    )
    assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0, \
        "FAILED: One of the splits is empty"
    assert len(train_idx) + len(val_idx) + len(test_idx) == ds.len(), \
        "FAILED: Splits don't sum to total samples"
    # No overlap
    assert set(train_idx).isdisjoint(val_idx), "FAILED: train/val overlap"
    assert set(val_idx).isdisjoint(test_idx),  "FAILED: val/test overlap"
    print(f"OK: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"    Test starts at: {test_start_ts.date()}")

    print("\n--- [TS-2] Short series Case B (<3 years): 67/33 split ---")
    seq_len = 96
    n_steps = 1 * 365 * 96        # only one year
    ts_short = pd.date_range("2023-01-01", periods=n_steps, freq="15min")

    ds2 = MagicMock(spec=EARNeGraphDataset)
    ds2.seq_len = seq_len
    ds2.timestamps = ts_short.tolist()
    ds2.len = lambda: n_steps - seq_len - 1

    with patch.object(EARNeGraphDataset, 'get_split_indices', EARNeGraphDataset.get_split_indices):
        train_idx2, val_idx2, test_idx2 = EARNeGraphDataset.get_split_indices(ds2)

    n2 = ds2.len()
    tv_end_expected = int(n2 * 0.67)
    assert test_idx2[0] == tv_end_expected, (
        f"FAILED: test_start={test_idx2[0]}, expected {tv_end_expected}"
    )
    assert len(train_idx2) + len(val_idx2) + len(test_idx2) == n2, \
        "FAILED: splits don't sum to total"
    frac_test = len(test_idx2) / n2
    assert 0.30 < frac_test < 0.36, f"FAILED: test fraction {frac_test:.3f} not near 33%"
    print(f"OK: train={len(train_idx2)}, val={len(val_idx2)}, test={len(test_idx2)}")
    print(f"    Test fraction: {frac_test:.3f} (expected ~0.33)")

    print("\n--- [TS-3] No-timestamp fallback: same 67/33 logic ---")
    ds3 = MagicMock(spec=EARNeGraphDataset)
    ds3.seq_len = seq_len
    ds3.timestamps = None   # simulate missing timestamps
    ds3.len = lambda: 1000

    with patch.object(EARNeGraphDataset, 'get_split_indices', EARNeGraphDataset.get_split_indices):
        train_idx3, val_idx3, test_idx3 = EARNeGraphDataset.get_split_indices(ds3)

    assert test_idx3[0] == int(1000 * 0.67), \
        f"FAILED: test_start={test_idx3[0]}, expected {int(1000 * 0.67)}"
    assert len(train_idx3) + len(val_idx3) + len(test_idx3) == 1000, \
        "FAILED: splits don't sum to 1000"
    print(f"OK: train={len(train_idx3)}, val={len(val_idx3)}, test={len(test_idx3)}")

    print("\n--- [TS-4] 2-Year boundary: falls back to Case B ---")
    n_steps_2y = 2 * 365 * 96
    ts_2y = pd.date_range("2022-01-01", periods=n_steps_2y, freq="15min")

    ds4 = MagicMock(spec=EARNeGraphDataset)
    ds4.seq_len = seq_len
    ds4.timestamps = ts_2y.tolist()
    ds4.len = lambda: n_steps_2y - seq_len - 1

    with patch.object(EARNeGraphDataset, 'get_split_indices', EARNeGraphDataset.get_split_indices):
        train_idx4, val_idx4, test_idx4 = EARNeGraphDataset.get_split_indices(ds4)

    n4 = ds4.len()
    # Should be Case B since only 2 years; test starts at 67%
    assert test_idx4[0] == int(n4 * 0.67), \
        f"FAILED: 2-year data should use Case B (67%), got test_start={test_idx4[0]}"
    print(f"OK: 2-year data correctly uses 67/33 fallback")
    print(f"    train={len(train_idx4)}, val={len(val_idx4)}, test={len(test_idx4)}")

    print("\nAll temporal split tests passed.")


if __name__ == "__main__":
    test_integrity()
    test_extended()
    test_temporal_split()
