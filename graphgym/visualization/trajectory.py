import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
from pprint import pprint

# --- 1. LATEX / THESIS STYLE CONFIG ---
def init_style():
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.bbox": "tight",
        "text.usetex": False, 
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "axes.spines.top": False,      
        "axes.spines.right": False,    
        "axes.linewidth": 0.5
    })

# --- 2. TRAJECTORY PLOTTING FUNCTION ---
def plot_trajectory(trajectories, metric_name, out_path, use_grid=False, mode="both"):
    # Using the user's aesthetic palette
    COLOR_MAP = ["#ECCD61", "#7FA3C2", "#B88FD5", "#34495e"]
    configs = sorted(trajectories.keys())
    n_configs = len(configs)

    if use_grid:
        cols = 2
        rows = (n_configs + 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows), sharex=True)
        axs = axs.flatten() if n_configs > 1 else [axs]
    else:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        axs = [ax] * n_configs

    for i, config in enumerate(configs):
        ax = axs[i]
        color = COLOR_MAP[i % len(COLOR_MAP)]

        # Plot Logic for Train/Val/Both
        modes_to_plot = ["train", "val"] if mode == "both" else [mode]

        for m in modes_to_plot:
            data = trajectories[config].get(m, [])
            if not data: continue

            # Sync lengths across seeds
            min_len = min(len(d) for d in data)
            cropped_data = np.array([d[:min_len] for d in data])

            start_epoch = 5
            if min_len > start_epoch:
                cropped_data = cropped_data[:, start_epoch:] # All seeds, epochs from 5 onwards
                epochs = np.arange(start_epoch, min_len)     # X-axis starts at 5
            else:
                epochs = np.arange(min_len)

            mean_vals = np.mean(cropped_data, axis=0)
            sem_vals = np.std(cropped_data, axis=0, ddof=1) / np.sqrt(len(data))

            ls = "-" if m == "val" else "--"
            alpha = 0.2 if m == "val" else 0.1
            label = f"{config[0]}|{config[1]} ({m})"

            ax.plot(epochs, mean_vals, label=label, color=color, linestyle=ls, linewidth=1.1)
            ax.fill_between(epochs, mean_vals - sem_vals, mean_vals + sem_vals, color=color, alpha=alpha)

        if use_grid:
            ax.set_title(f"{config[0]} | {config[1]}", fontsize=7, fontweight='bold')
            if i % 2 == 0: ax.set_ylabel(metric_name.upper())

    if not use_grid:
        # Complex legend for overlapping lines
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=2, frameon=False)

    fig.supxlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[✓] Plot saved to {out_path} (Mode: {mode})")

# --- 3. DATA LOADING ---
def load_trajectories(grid_dir, metric_name):
    trajectories = {}
    p = Path(grid_dir)
    
    for subdir in p.iterdir():
        if not subdir.is_dir() or '-' not in subdir.name: continue
            
        parts = subdir.name.split('-')
        weather, mp = "Unknown", "Unknown"
        for part in parts:
            if part.startswith('w='): weather = "Weather" if len(part[2:]) > 2 else "No Weather"
            if part.startswith('mp='): mp = f"MP={part[3:]}"
        
        config_key = (weather, mp)
        trajectories[config_key] = {"train": [], "val": []}
        
        for m in ["train", "val"]:
            for seed_dir in subdir.glob('[0-9]*'):
                stats_file = seed_dir / m / 'stats.json'
                if stats_file.exists():
                    history = []
                    with open(stats_file, 'r') as f:
                        for line in f:
                            try:
                                d = json.loads(line)
                                if metric_name in d: history.append(d[metric_name])
                            except: continue
                    if history: trajectories[config_key][m].append(history)
                    
    return trajectories

# --- 4. MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--metric', type=str, default='earne_mae_load')
    parser.add_argument('--out', type=str, default='trajectory.pdf')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--mode', choices=['train', 'val', 'both'], default='val')
    args = parser.parse_args()

    init_style()
    data = load_trajectories(args.dir, args.metric)
    
    if data:
        print("\n--- Verbose Data Check ---")
        pprint(data)
        plot_trajectory(data, args.metric, args.out, use_grid=args.grid, mode=args.mode)
    else:
        print("No data found.")
