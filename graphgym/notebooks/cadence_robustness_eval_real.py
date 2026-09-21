"""
cadence_robustness_eval_real.py

Real-data counterpart to cadence_robustness_eval.py -- Phase 0/2 of
~/.claude/plans/squishy-wobbling-bunny.md, now using genuinely real 1-minute
smartmeter data (see onemin_stream_adapter.py's module docstring for how
Phase 0's data-location blocker was resolved) instead of the linear-
interpolation placeholder.

Scope/caveat, stated once here rather than repeated everywhere: the one
raw household used below (Smartmeter-ID 08F9E07960A5-P1, paired Solar-ID
FDF49722 via meta_data.csv) is NOT confirmed to be one of the 120
households in the training population -- the device_id -> training
user_id mapping isn't recoverable from currently-available artifacts (see
onemin_stream_adapter.py docstring). This script therefore only runs
architectures with node-agnostic shared weights (MLP/LSTM/CVAE), where
running on a novel household is still a well-defined forward pass. GNN
(fixed training-population graph) and Linear/SVR/KNN (one fit per
training household) are not run here.

Usage:
    python notebooks/cadence_robustness_eval_real.py \
        --cfg configs/pyg/res_eval_lstm_5min_dualmask.yaml \
        --ckpt "results/res_eval_lstm_5min_dualmask/0/ckpt/epoch=16-step=117283.ckpt" \
        --arch lstm --device-id 08F9E07960A5-P1 --n-samples 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cadence_robustness_eval import FINE_LEN, load_model_and_dataset
from custom_graphgym.loader.graph_dataset import _encode_temporal
from custom_graphgym.metric.regression import DisaggregationMetrics
from custom_graphgym.target_utils import active_targets
from onemin_stream_adapter import (
    load_real_oneminute,
    load_real_pv_ground_truth,
    solar_id_for,
)

from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg

SEQ_LEN = 288  # matches cfg.model.seq_len for the *_5min_dualmask configs


def build_real_frames(device_id: str):
    """Real 5-min grid (consumption_w, generation_w, inverter_w -- all
    genuinely measured, no interpolation) + real native 1-min
    (consumption_w, generation_w) for the same device."""
    five_min = load_real_oneminute(device_id, cadence_minutes=5)
    solar_id = solar_id_for(device_id)
    pv_5min = load_real_pv_ground_truth(solar_id)
    merged = five_min.merge(pv_5min, on="timestamp", how="inner")

    one_min = load_real_oneminute(device_id, cadence_minutes=1)
    return merged, one_min


def find_valid_windows(merged: pd.DataFrame, one_min: pd.DataFrame, max_windows: int):
    """Reindex onto a strict 5-min grid, then find target indices with
    288 consecutive real, non-null 5-min rows AND 5 consecutive real,
    non-null 1-min rows immediately preceding the target -- no filling,
    no interpolation, genuinely real data only in both places."""
    merged = merged.sort_values("timestamp").set_index("timestamp")
    full_index = pd.date_range(merged.index.min(), merged.index.max(), freq="5min")
    grid = merged.reindex(full_index)
    valid = grid[["consumption_w", "generation_w", "inverter_w"]].notna().all(axis=1).to_numpy()

    one_min = one_min.sort_values("timestamp").set_index("timestamp")
    one_min_valid = one_min[["consumption_w", "generation_w"]].notna().all(axis=1)

    candidates = []
    n = len(valid)
    running = 0
    for i in range(n):
        running = running + 1 if valid[i] else 0
        if running >= SEQ_LEN:
            tgt_time = full_index[i]
            fine_times = pd.date_range(tgt_time - pd.Timedelta(minutes=4), tgt_time, freq="1min")
            if all(t in one_min_valid.index and one_min_valid.loc[t] for t in fine_times):
                candidates.append(i)

    if len(candidates) > max_windows:
        idx = np.linspace(0, len(candidates) - 1, num=max_windows, dtype=int)
        candidates = [candidates[j] for j in idx]
    return grid, full_index, one_min, candidates


def build_control_and_mixed(grid, full_index, one_min, i, transform_obj, dim_in):
    window = grid.iloc[i - SEQ_LEN + 1 : i + 1]
    tgt_time = full_index[i]

    cons_raw = torch.tensor(window["consumption_w"].to_numpy(), dtype=torch.float32)
    gen_raw = torch.tensor(window["generation_w"].to_numpy(), dtype=torch.float32)
    pv_raw = float(window["inverter_w"].iloc[-1])

    cons_scaled = transform_obj.transform("consumption", cons_raw)
    gen_scaled = transform_obj.transform("generation", gen_raw) if dim_in == 2 else None

    if dim_in == 2:
        control_x = torch.stack([cons_scaled, gen_scaled], dim=-1).unsqueeze(0)  # [1, 288, 2]
    else:
        control_x = cons_scaled.unsqueeze(0).unsqueeze(-1)  # [1, 288, 1]

    fine_times = pd.date_range(tgt_time - pd.Timedelta(minutes=FINE_LEN - 1), tgt_time, freq="1min")
    fine_cons_raw = torch.tensor(
        one_min.loc[fine_times, "consumption_w"].to_numpy(), dtype=torch.float32
    )
    fine_gen_raw = torch.tensor(
        one_min.loc[fine_times, "generation_w"].to_numpy(), dtype=torch.float32
    )
    fine_cons_scaled = transform_obj.transform("consumption", fine_cons_raw)

    mixed_x = control_x.clone()
    mixed_x[0, -FINE_LEN:, 0] = fine_cons_scaled
    if dim_in == 2:
        # generation IS also 1-min-native at the raw source (EXPORT_KW) --
        # see onemin_stream_adapter.py's docstring correction -- so refresh
        # it too, unlike the synthetic-only harness which held it constant.
        fine_gen_scaled = transform_obj.transform("generation", fine_gen_raw)
        mixed_x[0, -FINE_LEN:, 1] = fine_gen_scaled

    temporal_full = _encode_temporal(pd.DatetimeIndex(window.index)).unsqueeze(0)
    fine_temporal = _encode_temporal(pd.DatetimeIndex(fine_times)).unsqueeze(0)
    mixed_temporal = torch.cat([temporal_full[:, :-FINE_LEN], fine_temporal], dim=1)

    operational = torch.ones(1, SEQ_LEN, 1)  # actively reporting throughout -- see script docstring

    y_pv_scaled = transform_obj.transform("pv", torch.tensor([pv_raw], dtype=torch.float32))
    y_load_scaled = torch.zeros(1)  # load target unused (predict_targets == ["pv"])
    y_net_demand_scaled = torch.zeros(1)  # unused when physics_weight == 0 at eval time
    mask = torch.ones(1)

    control = Data(
        x=control_x, temporal=temporal_full, operational=operational,
        y_load=y_load_scaled, y_pv=y_pv_scaled, y_net_demand=y_net_demand_scaled,
        mask=mask, num_nodes=1,
        edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 1)),
    )
    mixed = Data(
        x=mixed_x, temporal=mixed_temporal, operational=operational,
        y_load=y_load_scaled, y_pv=y_pv_scaled, y_net_demand=y_net_demand_scaled,
        mask=mask, num_nodes=1,
        edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 1)),
    )
    return control, mixed


def run(cfg_path: str, ckpt_path: str, arch: str, device_id: str, n_samples: int):
    dataset, model, device = load_model_and_dataset(cfg_path, ckpt_path)
    transform_obj = dataset.transform_obj
    dim_in = cfg.model.dim_in
    targets = active_targets()
    n_q = cfg.model.n_quantiles

    merged, one_min = build_real_frames(device_id)
    grid, full_index, one_min_indexed, candidates = find_valid_windows(merged, one_min, n_samples)
    print(f"real valid windows found: {len(candidates)} (requested up to {n_samples})")
    if not candidates:
        raise RuntimeError("No valid real windows found -- check data coverage for this device.")

    results = {"control_5min": ([], []), "mixed_cadence": ([], [])}
    with torch.no_grad():
        for i in candidates:
            control, mixed = build_control_and_mixed(
                grid, full_index, one_min_indexed, i, transform_obj, dim_in
            )
            for cond, data in (("control_5min", control), ("mixed_cadence", mixed)):
                batch = Batch.from_data_list([data]).to(device)
                pred, true = model(batch)
                results[cond][0].append(true.cpu())
                results[cond][1].append(pred.cpu())

    rows = []
    for cond, (true_chunks, pred_chunks) in results.items():
        true = torch.stack(true_chunks, dim=-1)
        pred = torch.stack(pred_chunks, dim=-1)
        pred_denorm = torch.cat(
            [transform_obj.inverse_transform(name, pred[:, i * n_q:(i + 1) * n_q])
             for i, name in enumerate(targets)], dim=1,
        )
        true_denorm = torch.cat(
            [transform_obj.inverse_transform("load", true[:, [0]]),
             transform_obj.inverse_transform("pv", true[:, [1]]),
             true[:, [2]],
             transform_obj.inverse_transform("net_demand", true[:, [3]])], dim=1,
        )
        metrics = DisaggregationMetrics.all(
            true_denorm.numpy(), pred_denorm.numpy(),
            targets=targets, n_quantiles=n_q, q50_idx=cfg.model.quantiles.index(0.5),
        )
        for target, target_metrics in metrics.items():
            for metric, value in target_metrics.items():
                rows.append({
                    "arch": arch, "condition": cond, "mitigation": "none",
                    "device": device.type, "data_source": "real_p1", "target": target,
                    "metric": metric, "value": value,
                })

    out = pd.DataFrame(rows)
    out_path = Path("notebooks/cadence_robustness_long.csv")
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    if not existing.empty:
        if "data_source" not in existing.columns:
            existing["data_source"] = "synthetic"
        existing = existing[~((existing["arch"] == arch) & (existing["data_source"] == "real_p1"))]
    out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(out)} rows total, {len(rows)} from arch={arch}, data_source=real_p1)")

    piv = out[(out["arch"] == arch) & (out.get("data_source") == "real_p1")].pivot_table(
        index="condition", columns="metric", values="value"
    )
    cols = [c for c in ["rmse", "nrmse", "mae", "nmae", "r2", "coverage", "sharpness"] if c in piv.columns]
    print(piv[cols].round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    run(args.cfg, args.ckpt, args.arch, args.device_id, args.n_samples)
