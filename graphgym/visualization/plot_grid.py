import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
from pprint import pprint # Added pprint for verbose output

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
        "figure.figsize": (3.5, 2.8), 
        "figure.dpi": 300,
        "axes.spines.top": False,      
        "axes.spines.right": False,    
        "axes.linewidth": 0.5
    })

# --- 2. GROUPED BARGRAPH FUNCTION ---
def grouped_bar(scores, x_label="Dataset", y_label="MAE", legend_title=None):
    COLOR = ["#EEEFEB", "#ECCD61", "#7FA3C2", "#B88FD5"]
    PATCH = [None, "oooo", None, "xxxx"]

    groups = sorted(list(set(x[0] for x in scores.keys())))
    bars_per_group = sorted(list(set(x[1] for x in scores.keys())))
    
    n_groups = len(groups)
    n_bars = len(bars_per_group)

    cfg = {
        "bar_width": 0.8 / (n_bars + 1),
        "group_gap": 0.4,
        "err_capsize": 1.5,
        "err_linewidth": 0.7,
        "label_padding": 2,
        "label_fontsize": 5,
        "legend_bbox": (0.5, 1.18),
        "edge_width": 0.5
    }
    
    fig, ax = plt.subplots()
    
    max_val = 0
    for i, bar_name in enumerate(bars_per_group):
        means, stderrs, positions = [], [], []
        
        for j, group_name in enumerate(groups):
            vals = scores.get((group_name, bar_name), [0])
            means.append(np.mean(vals))
            stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            
            pos = j * (n_bars * cfg["bar_width"] + cfg["group_gap"]) + (i * cfg["bar_width"])
            positions.append(pos)

        container = ax.bar(
            positions, means, cfg["bar_width"],
            yerr=stderrs, 
            label=bar_name,
            edgecolor="black", 
            linewidth=cfg["edge_width"], 
            capsize=cfg["err_capsize"],
            error_kw={'elinewidth': cfg['err_linewidth']},
            hatch=PATCH[i % len(PATCH)], 
            color=COLOR[i % len(COLOR)]
        )
        
        ax.bar_label(container, fmt='%.1f', padding=cfg["label_padding"], fontsize=cfg["label_fontsize"])
        max_val = max(max_val, max(means) if means else 0)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    tick_positions = [
        j * (n_bars * cfg["bar_width"] + cfg["group_gap"]) + (cfg["bar_width"] * (n_bars - 1) / 2)
        for j in range(n_groups)
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, max_val * 1.3)
    
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=cfg["legend_bbox"], 
        ncols=min(n_bars, 4), 
        frameon=False,
        title=legend_title
    )
    
    return fig

# --- 3. DATA LOADING ---
def load_metrics(grid_dir, metric_name):
    results = {}
    p = Path(grid_dir)
    
    for subdir in p.iterdir():
        if not subdir.is_dir() or '-' not in subdir.name:
            continue
            
        parts = subdir.name.split('-')
        weather = "Unknown"
        mp = "Unknown"
        
        for part in parts:
            if part.startswith('w='):
                val = part[2:]
                weather = "Weather" if len(val) > 2 else "No Weather"
            if part.startswith('mp='):
                mp = f"MP={part[3:]}"
        
        scores = []
        for seed_dir in subdir.glob('[0-9]*'):
            stats_file = seed_dir / 'val' / 'stats.json'
            if stats_file.exists():
                print(f"Reading stats file: {stats_file}")
                with open(stats_file, 'r') as f:
                    best_val = float('inf')
                    for line in f:
                        try:
                            data = json.loads(line)
                            if metric_name in data:
                                best_val = min(best_val, data[metric_name])
                        except:
                            continue
                    if best_val != float('inf'):
                        scores.append(best_val)
        
        if scores:
            results[(weather, mp)] = scores
            
    return results

# --- 4. MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Results grid directory')
    parser.add_argument('--metric', type=str, default='earne_mae_load', help='Metric to plot')
    parser.add_argument('--out', type=str, default='plot.pdf', help='Output filename')
    args = parser.parse_args()

    init_style()
    data = load_metrics(args.dir, args.metric)
    
    if not data:
        print(f"\n[!] No data found for metric '{args.metric}' in {args.dir}")
    else:
        # --- VERBOSE OUTPUT ---
        print(f"\n--- Aggregated Data for Metric: {args.metric} ---")
        pprint(data)
        print("\n--- Summary ---")
        for key, vals in data.items():
            print(f"{key}: {len(vals)} seed(s) found. Mean: {np.mean(vals):.4f}")
        print("----------------\n")

        label = args.metric.replace('earne_mae_', '').upper()
        fig = grouped_bar(data, x_label="Data Configuration", y_label=f"{label} MAE")
        plt.savefig(args.out)
        print(f"[✓] Plot saved to {args.out}")
