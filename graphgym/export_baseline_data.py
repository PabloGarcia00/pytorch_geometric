import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Mock GraphGym config to get split ratios (should match earne_test_refactored.yaml)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.85
SEQ_LEN = 96

def export_data(bundle_path, out_dir):
    print(f"Loading bundle from {bundle_path}...")
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    
    net = bundle['net_scaled']      # [T, N]
    load = bundle['load_scaled']    # [T, N]
    pv = bundle['pv_scaled']        # [T, N]
    weather = bundle['weather_data']# [T, N, W]
    temporal = bundle['temporal_data'] # [T, 6]
    mask = bundle['mask_raw']       # [T, N]
    
    T, N = net.shape
    num_samples = T - SEQ_LEN
    
    # Identify splits
    tr_end = int(TRAIN_SPLIT * num_samples)
    val_end = int(VAL_SPLIT * num_samples)
    
    splits = {
        'train': range(0, tr_end),
        'val': range(tr_end, val_end),
        'test': range(val_end, num_samples)
    }
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    for split_name, indices in splits.items():
        print(f"Processing {split_name} split...")
        rows = []
        for i in indices:
            target_idx = i + SEQ_LEN
            
            # Global temporal features for this step
            t_feat = temporal[target_idx].numpy()
            
            for n in range(N):
                if mask[target_idx, n] == 0:
                    continue
                    
                # Node-specific historical window (last SEQ_LEN steps)
                # We'll use the mean/max of history as simple baseline features
                history = net[i : target_idx, n].numpy()
                h_mean = history.mean()
                h_max = history.max()
                h_last = history[-1]
                
                # Current weather for this node
                w_feat = weather[target_idx, n].numpy()
                
                # Current Net Demand (Input)
                current_net = net[target_idx, n].item()
                
                # Targets
                target_load = load[target_idx, n].item()
                target_pv = pv[target_idx, n].item()
                
                row = np.concatenate([
                    [n, current_net], 
                    [h_mean, h_max, h_last],
                    t_feat, 
                    w_feat, 
                    [target_load, target_pv]
                ])
                rows.append(row)
        
        # Define Columns
        columns = ['node_id', 'net_demand', 'h_mean', 'h_max', 'h_last']
        columns += [f'temp_{j}' for j in range(6)]
        columns += [f'weather_{j}' for j in range(weather.shape[2])]
        columns += ['target_load', 'target_pv']
        
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(f"{out_dir}/{split_name}.csv", index=False)
        print(f"Saved {split_name}.csv with {len(df)} samples.")

if __name__ == "__main__":
    bundle_file = 'datasets/earne/processed/earne_master_spatial_knn.pt'
    output_directory = 'baseline_data'
    export_data(bundle_file, output_directory)
