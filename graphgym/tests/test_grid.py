"""
EARNe Configuration Test Grid
==============================
Tests all major config axes in a single run:

  Graph modes  : spatial_knn | full_graph | learned_corr
  Norm modes   : minmax      | zero_log
  Dual-read    : False (single net-demand channel) | True (import + export)
  Split cases  : auto-detected (Case A ≥3 yrs / Case B short-series)

Each cell does the minimum needed to surface a real failure:
  1. Dataset loads (normalisation, topology, split)
  2. Split sizes look sane (no empty train/test)
  3. One forward pass through the model
  4. Loss computes a finite value

Run from the graphgym/ directory:
    python tests/test_grid.py
    python tests/test_grid.py 2>&1 | tee test_grid.log
"""

import os
import sys
import traceback
from itertools import product

sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from torch_geometric.graphgym.config import cfg, load_cfg
import custom_graphgym  # noqa – registers all custom modules

# ------------------------------------------------------------------ #
# Grid axes
# ------------------------------------------------------------------ #
GRAPH_MODES  = ['spatial_knn', 'full_graph', 'learned_corr']
NORM_MODES   = ['minmax', 'zero_log']
# dual_read=True requires dim_in=2; paired to avoid invalid combos
DUAL_CONFIGS = [(False, 1), (True, 2)]   # (dual_read, dim_in)
GRID         = list(product(GRAPH_MODES, NORM_MODES, DUAL_CONFIGS))

BASE_CFG    = 'configs/test_compat/earne_weather_all_15min_grid_test_compat/' \
              'earne_weather_all_15min-graph_mode=spatial_knn-norm_mode=minmax-days=3-n_user=50-epoch=1.yaml'

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
COL_W = 18

def _header(graph_mode, norm_mode, dual_read):
    dr = 'dual' if dual_read else 'single'
    return f"[{graph_mode:14s} | {norm_mode:7s} | {dr}]"

def _load_base_cfg(graph_mode, norm_mode, dual_read, dim_in):
    """Reset cfg to defaults then override all three axes."""
    load_cfg(cfg, type('Args', (), {'cfg_file': BASE_CFG, 'opts': []})())
    cfg.accelerator = 'cpu'
    cfg.earne_data.graph_mode = graph_mode
    cfg.earne_data.norm_mode  = norm_mode
    cfg.earne_data.dual_read  = dual_read
    cfg.model.dim_in          = dim_in
    # Keep days=3 so the test runs on a tiny slice
    cfg.earne_data.days = 3

def _check_splits(dataset, label):
    """Assert all three splits are non-empty and don't overlap."""
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    n = dataset.len()
    issues = []
    if len(train_idx) == 0: issues.append("train is empty")
    if len(val_idx)   == 0: issues.append("val is empty")
    if len(test_idx)  == 0: issues.append("test is empty")
    total = len(train_idx) + len(val_idx) + len(test_idx)
    if total != n:
        issues.append(f"splits sum to {total} but len={n}")
    if set(train_idx) & set(val_idx):
        issues.append("train/val overlap")
    if set(val_idx) & set(test_idx):
        issues.append("val/test overlap")
    return train_idx, val_idx, test_idx, issues

def _forward_pass(dataset):
    """One batch forward pass; returns (pred_shape, loss_value)."""
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.loader import DataLoader

    model = create_model()
    model.eval()

    # Use the first few samples for a single test batch
    sample_indices = list(range(min(4, dataset.len())))
    batch_list = [dataset.get(i) for i in sample_indices]

    from torch_geometric.data import Batch
    batch = Batch.from_data_list(batch_list)

    with torch.no_grad():
        pred, true = model(batch)

    # Compute loss
    from custom_graphgym.loss.earne_loss import earne_loss_complex
    loss, _ = earne_loss_complex(pred, true)

    return pred.shape, float(loss)


# ------------------------------------------------------------------ #
# Main runner
# ------------------------------------------------------------------ #
def run_grid():
    if not os.path.exists(BASE_CFG):
        print(f"ERROR: Base config not found: {BASE_CFG}")
        return

    results = {}  # (graph_mode, norm_mode, dual_read) -> dict

    for graph_mode, norm_mode, (dual_read, dim_in) in GRID:
        label = _header(graph_mode, norm_mode, dual_read)
        print(f"\n{'='*70}")
        print(f" {label}")
        print(f"{'='*70}")
        cell = {'status': 'PASS', 'notes': [], 'dual_read': dual_read}

        try:
            # ---- Step 1: Config ----
            _load_base_cfg(graph_mode, norm_mode, dual_read, dim_in)
            print(f"  [1/4] Config set  (dual_read={dual_read}, dim_in={dim_in})")

            # ---- Step 2: Dataset load + normalisation ----
            from custom_graphgym.loader.graph_dataset import EARNeGraphDataset
            dataset = EARNeGraphDataset(
                root=cfg.earne_data.processed_root,
                seq_len=cfg.model.seq_len,
            )
            print(f"  [2/4] Dataset loaded — T={dataset.limit_t}, N={dataset.num_nodes}, "
                  f"samples={dataset.len()}")

            # Spot-check dual_read tensor shape
            sample = dataset.get(0)
            expected_channels = 2 if dual_read else 1
            if sample.x.shape[-1] != expected_channels:
                cell['notes'].append(
                    f"x.shape[-1]={sample.x.shape[-1]}, expected {expected_channels}"
                )
                cell['status'] = 'WARN'
            else:
                print(f"        x shape: {tuple(sample.x.shape)}  ✓")

            # ---- Step 3: Splits ----
            train_idx, val_idx, test_idx, split_issues = _check_splits(dataset, label)
            if split_issues:
                cell['notes'].extend(split_issues)
                cell['status'] = 'WARN'
            split_case = 'Case A (3yr)' if len(set(
                [__import__('pandas').Timestamp(t).year
                 for t in (dataset.timestamps or [])[dataset.seq_len:]]
            )) >= 3 else 'Case B (short)'
            print(f"  [3/4] Splits OK — {split_case} | "
                  f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

            # ---- Step 4: Forward pass + loss ----
            pred_shape, loss_val = _forward_pass(dataset)
            if not np.isfinite(loss_val):
                cell['notes'].append(f"non-finite loss: {loss_val}")
                cell['status'] = 'WARN'
            print(f"  [4/4] Forward pass OK — pred={pred_shape}, loss={loss_val:.6f}")

            cell['pred_shape'] = str(pred_shape)
            cell['loss']       = f"{loss_val:.4f}"
            cell['split_case'] = split_case
            cell['n_train']    = len(train_idx)
            cell['n_test']     = len(test_idx)

        except Exception as e:
            cell['status'] = 'FAIL'
            cell['error']  = str(e)
            cell['trace']  = traceback.format_exc()
            print(f"  FAILED: {e}")
            print(cell['trace'])

        results[(graph_mode, norm_mode, dual_read)] = cell

    # ---- Summary table ----
    print(f"\n\n{'='*84}")
    print(" GRID SUMMARY")
    print(f"{'='*84}")
    print(f"{'Graph mode':<16} {'Norm':<9} {'DualRd':<8} {'Status':<8} "
          f"{'Loss':<10} {'Split':<16} {'Notes'}")
    print(f"{'-'*84}")
    for (gm, nm, dr), c in results.items():
        status = c['status']
        loss   = c.get('loss', '—')
        split  = c.get('split_case', '—')
        notes  = '; '.join(c.get('notes', [])) or (c.get('error', '')[:36] if status == 'FAIL' else '—')
        dr_s   = 'yes' if dr else 'no'
        print(f"{gm:<16} {nm:<9} {dr_s:<8} {status:<8} {loss:<10} {split:<16} {notes}")
    print(f"{'='*84}\n")

    n_pass = sum(1 for c in results.values() if c['status'] == 'PASS')
    n_warn = sum(1 for c in results.values() if c['status'] == 'WARN')
    n_fail = sum(1 for c in results.values() if c['status'] == 'FAIL')
    print(f"Results: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL  "
          f"(total {len(GRID)})")


if __name__ == '__main__':
    run_grid()
