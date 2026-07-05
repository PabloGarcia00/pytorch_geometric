"""
simulation.py — Run the trained model over the entire test split.

Set EXP_DIR below, then run:
    %run visualization/simulation.py
or
    python visualization/simulation.py

After running, the following are available in the namespace:

Predictions (normalized, index=timestamps, MultiIndex cols=(mac, q_label)):
    df_pred_load_norm
    df_pred_pv_norm

Ground-truth / input (normalized, index=timestamps, columns=macs):
    df_gt_load_norm
    df_gt_pv_norm
    df_gt_net_norm

Helpers:
    denorm_df(df)             -> denormalized copy in Watts  (uses p1, p2, denorm_fn)
    pivot_to_daily(df)        -> (date × mac) × t_00..t_95 + metadata

Also exposed:
    full_ds, model, device, p1, p2, denorm_fn
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "graphgym"))

import custom_graphgym  # noqa
from custom_graphgym.loader.normalization import (
    denorm_ihs,
    denorm_log1p,
    denorm_minmax,
    denorm_zscore,
)

from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model

# ---------------------------------------------------------------------------
# ★ Configure here
# ---------------------------------------------------------------------------

EXP_DIR = "results/earne_exp2_grid_exp2/earne_exp2-weather_mode=False-mp=0-graph_mode=full_graph-norm_mode=ihs"

# ---------------------------------------------------------------------------

_DENORM = {
    "ihs": denorm_ihs,
    "minmax": denorm_minmax,
    "log1p": denorm_log1p,
    "znorm": denorm_zscore,
}

STEPS_PER_DAY = 96


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def unwrap_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _find_config(exp_path: Path) -> Path:
    for candidate in [
        exp_path / "config.yaml",
        exp_path.parent / "config.yaml",
    ]:
        if candidate.exists():
            return candidate
    configs = list(exp_path.glob("**/config.yaml"))
    if configs:
        return configs[0]
    raise FileNotFoundError(f"No config.yaml found under {exp_path}")


def _find_latest_checkpoint(exp_path: Path) -> Path:
    ckpt_dir = exp_path / "ckpt"
    ckpts = (
        list(ckpt_dir.glob("*.ckpt"))
        if ckpt_dir.exists()
        else list(exp_path.glob("**/ckpt/*.ckpt"))
    )
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {exp_path}")
    return sorted(ckpts, key=os.path.getmtime)[-1]


def _load_model(ckpt: Path, device: torch.device):
    model = create_model()
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint.get(
        "state_dict", checkpoint.get("model_state_dict", checkpoint)
    )
    mk = list(model.state_dict().keys())
    ck = list(state_dict.keys())
    if mk[0].startswith("model.") and not ck[0].startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif ck[0].startswith("model.") and not mk[0].startswith("model."):
        state_dict = {
            k.replace("model.", "", 1): v for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _load_norm_params():
    e = cfg.earne_data.energy_norm_mode
    w = cfg.earne_data.weather_norm_mode
    g = cfg.earne_data.graph_mode
    path = Path(cfg.dataset.dir) / f"norm_params_{e}_{w}_{g}.pt"
    raw = torch.load(path, weights_only=False)
    return raw["p1"], raw["p2"], e


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulation(model, dataset, device):
    """Inference over every index in *dataset*.

    Returns
    -------
    df_pred_load_norm : MultiIndex cols (mac, q_label)
    df_pred_pv_norm   : same
    df_gt_load_norm   : cols = macs
    df_gt_pv_norm     : cols = macs
    df_gt_net_norm    : cols = macs  (net demand at target step)
    """
    full_ds = unwrap_dataset(dataset)
    n_q = cfg.model.n_quantiles
    q_labels = [f"q{int(round(q * 100)):02d}" for q in cfg.model.quantiles]
    macs = full_ds.active_macs
    N = full_ds.num_nodes

    indices = (
        list(
            dataset.indices() if callable(dataset.indices) else dataset.indices
        )
        if hasattr(dataset, "indices")
        else list(range(len(dataset)))
    )
    T = len(indices)
    print(f"Simulating {T} timesteps × {N} nodes × {n_q} quantiles …")

    pred_load_buf = np.empty((T, N, n_q), dtype=np.float32)
    pred_pv_buf = np.empty((T, N, n_q), dtype=np.float32)
    gt_load_buf = np.empty((T, N), dtype=np.float32)
    gt_pv_buf = np.empty((T, N), dtype=np.float32)
    gt_net_buf = np.empty((T, N), dtype=np.float32)
    timestamps = []

    with torch.no_grad():
        for step, idx in enumerate(indices):
            if step % 200 == 0:
                print(f"  {step:>6}/{T}", end="\r", flush=True)

            batch = full_ds.get(idx).to(device)
            pred, _ = model(batch)  # [N, 2*n_q]

            pred_load_buf[step] = pred[:, :n_q].cpu().numpy()
            pred_pv_buf[step] = pred[:, n_q : 2 * n_q].cpu().numpy()
            gt_load_buf[step] = batch.y_load.cpu().numpy()
            gt_pv_buf[step] = batch.y_pv.cpu().numpy()
            gt_net_buf[step] = batch.y_net_demand.cpu().numpy()
            timestamps.append(
                batch.timestamp
                if (hasattr(batch, "timestamp") and batch.timestamp is not None)
                else idx
            )

    print(f"\nBuilding DataFrames …")
    ts_idx = pd.Index(pd.to_datetime(timestamps), name="timestamp")

    df_gt_load_norm = pd.DataFrame(gt_load_buf, index=ts_idx, columns=macs)
    df_gt_pv_norm = pd.DataFrame(gt_pv_buf, index=ts_idx, columns=macs)
    df_gt_net_norm = pd.DataFrame(gt_net_buf, index=ts_idx, columns=macs)

    mi = pd.MultiIndex.from_product([macs, q_labels], names=["mac", "quantile"])
    df_pred_load_norm = pd.DataFrame(
        pred_load_buf.reshape(T, N * n_q), index=ts_idx, columns=mi
    )
    df_pred_pv_norm = pd.DataFrame(
        pred_pv_buf.reshape(T, N * n_q), index=ts_idx, columns=mi
    )

    return (
        df_pred_load_norm,
        df_pred_pv_norm,
        df_gt_load_norm,
        df_gt_pv_norm,
        df_gt_net_norm,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def denorm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a denormalized copy in Watts (uses module-level p1, p2, denorm_fn)."""
    values = torch.tensor(df.values, dtype=torch.float32)
    out = denorm_fn(values, p1, p2).numpy()
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def pivot_to_daily(
    df: pd.DataFrame, value_name: str = "value", include_weather: bool = True
) -> pd.DataFrame:
    """Reshape a timestamps × nodes DataFrame into a day-level table.

    Each row = one (date, mac). Columns:
      t_00 … t_95      96 intra-day values (15-min slots)
      lat, lon, zip    node metadata
      w_<feat>_mean/std  daily aggregated weather (when include_weather=True)

    For prediction DFs with MultiIndex columns, select a quantile first:
        pivot_to_daily(df_pred_load_norm.xs("q50", level="quantile", axis=1))
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    df_long = df.reset_index().melt(
        id_vars="timestamp", var_name="mac", value_name=value_name
    )
    df_long["date"] = df_long["timestamp"].dt.normalize()
    df_long["slot"] = (
        df_long["timestamp"].dt.hour * 4 + df_long["timestamp"].dt.minute // 15
    )

    df_pivot = (
        df_long.pivot_table(
            index=["date", "mac"], columns="slot", values=value_name
        )
        .rename(columns={s: f"t_{s:02d}" for s in range(STEPS_PER_DAY)})
        .reset_index()
    )

    pos_np = (
        full_ds.pos.numpy()
        if torch.is_tensor(full_ds.pos)
        else np.array(full_ds.pos)
    )
    meta = pd.DataFrame(
        {
            "mac": full_ds.active_macs,
            "lat": pos_np[:, 0],
            "lon": pos_np[:, 1],
            "zip": full_ds.active_zips,
        }
    )
    df_pivot = df_pivot.merge(meta, on="mac", how="left")

    if include_weather and getattr(full_ds, "weather_data", None) is not None:
        feat_names = cfg.earne_data.weather_features
        w_np = full_ds.weather_data.numpy()  # [T_all, N, W_master]
        w_sel = w_np[:, :, full_ds.weather_idx]  # [T_all, N, W_req]
        all_ts = pd.to_datetime(full_ds.timestamps)

        ts_set = set(df.index)
        t_mask = np.array([t in ts_set for t in all_ts])
        w_sel_test = w_sel[t_mask]
        ts_test = all_ts[t_mask]

        rows_w = []
        for node_i, mac in enumerate(full_ds.active_macs):
            node_w = pd.DataFrame(
                w_sel_test[:, node_i, :], index=ts_test, columns=feat_names
            )
            daily = (
                node_w.resample("D").mean().add_prefix("w_").add_suffix("_mean")
            )
            daily = daily.join(
                node_w.resample("D").std().add_prefix("w_").add_suffix("_std")
            )
            daily["mac"] = mac
            daily.index.name = "date"
            rows_w.append(daily.reset_index())

        df_weather = pd.concat(rows_w, ignore_index=True)
        df_pivot = df_pivot.merge(df_weather, on=["date", "mac"], how="left")

    slot_cols = [f"t_{s:02d}" for s in range(STEPS_PER_DAY)]
    weather_cols = [c for c in df_pivot.columns if c.startswith("w_")]
    ordered = ["date", "mac", "lat", "lon", "zip"] + slot_cols + weather_cols
    return df_pivot[[c for c in ordered if c in df_pivot.columns]]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

exp_path = Path(EXP_DIR)
config_path = _find_config(exp_path)
ckpt_path = _find_latest_checkpoint(exp_path)

print(f"Config    : {config_path}")
print(f"Checkpoint: {ckpt_path}")

cfg.set_new_allowed(True)

import argparse

load_cfg(cfg, argparse.Namespace(cfg_file=str(config_path), opts=[]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

model = _load_model(ckpt_path, device)

loaders = create_loader()
next(iter(loaders[0]))["x"]
full_ds = unwrap_dataset(loaders[2].dataset)

p1, p2, emode = _load_norm_params()
denorm_fn = _DENORM[emode]

(
    df_pred_load_norm,
    df_pred_pv_norm,
    df_gt_load_norm,
    df_gt_pv_norm,
    df_gt_net_norm,
) = run_simulation(model, loaders[2].dataset, device)

print("\n--- Available ---")
for name, df in [
    ("df_pred_load_norm", df_pred_load_norm),
    ("df_pred_pv_norm", df_pred_pv_norm),
    ("df_gt_load_norm", df_gt_load_norm),
    ("df_gt_pv_norm", df_gt_pv_norm),
    ("df_gt_net_norm", df_gt_net_norm),
]:
    print(f"  {name:25s}  {df.shape}")
print("\ndenorm_df(df)       → Watts")
print("pivot_to_daily(df)  → (date × mac) × t_00..t_95 + metadata")
