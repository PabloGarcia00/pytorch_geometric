"""Training script for EARNe baselines."""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd

# Add paths to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "baselines"))
sys.path.insert(0, str(project_root / "graphgym"))

from models.persistence import PersistenceBaseline
from models.historical_mean import HistoricalMeanBaseline
from models.mlp import MLPBaseline
from BiLSTM import BiLSTMBaseline
from CVAE import CVAEBaseline

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(data_root: str, seq_len: int, train_split: float, val_split: float):
    # Try multiple path resolutions for the bundle file
    data_root_path = Path(data_root)
    bundle_path = data_root_path / "processed" / "earne_master_bundle.pt"
    if not bundle_path.exists():
        bundle_path = data_root_path / "processed" / "data.pt"
    if not bundle_path.exists():
        bundle_path = data_root_path / "earne_master_bundle.pt"
    if not bundle_path.exists():
        bundle_path = data_root_path / "data.pt"
        
    if not bundle_path.exists():
        # Try relative to the script location
        base_path = Path(__file__).resolve().parent
        bundle_path = base_path / data_root / "processed" / "earne_master_bundle.pt"
        if not bundle_path.exists():
            bundle_path = base_path / data_root / "processed" / "data.pt"

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Could not find processed bundle file (earne_master_bundle.pt or data.pt) in {data_root}"
        )

    print(f"Loading data from: {bundle_path}")
    master = torch.load(bundle_path, map_location="cpu", weights_only=False)

    ids = master["ids"]
    zips = master["zips"]
    mask_data = master["mask"]
    timestamps = master["timestamps"]

    # Apply node filtering (identical to EARNeGraphDataset)
    # Require at least 3 distinct calendar years of observed data
    ts_years = pd.to_datetime(timestamps).year.values
    year_mask = torch.tensor(
        [
            len(set(ts_years[mask_data[:, i].bool().numpy()])) >= 3
            for i in range(len(ids))
        ],
        dtype=torch.bool,
    )
    node_indices = torch.where(year_mask)[0]
    print(f"Filtered nodes: {len(node_indices)} / {len(ids)}")

    load_scaled = master["load_scaled"][:, node_indices]
    pv_scaled = master["pv_scaled"][:, node_indices]
    net_scaled = master["net_scaled"][:, node_indices]
    mask_data = mask_data[:, node_indices]
    operational = master["operational"][:, node_indices]

    T = net_scaled.shape[0]
    S = T - seq_len - 1

    # Slide window to build (x, y_load, y_pv)
    # x: [S, N, seq_len]
    # unfold along time axis (dim 0)
    x = net_scaled[:-1].unfold(0, seq_len, 1)[:S]
    y_load = load_scaled[seq_len : seq_len + S]
    y_pv = pv_scaled[seq_len : seq_len + S]
    target_timestamps = timestamps[seq_len : seq_len + S]
    target_mask = mask_data[seq_len : seq_len + S]
    target_operational = operational[seq_len : seq_len + S]

    # Chronological splits
    train_end = int(S * train_split)
    val_end = int(S * val_split)

    splits = {
        "train": {
            "x": x[:train_end],
            "y_load": y_load[:train_end],
            "y_pv": y_pv[:train_end],
            "timestamps": target_timestamps[:train_end],
            "mask": target_mask[:train_end],
            "operational": target_operational[:train_end],
        },
        "val": {
            "x": x[train_end:val_end],
            "y_load": y_load[train_end:val_end],
            "y_pv": y_pv[train_end:val_end],
            "timestamps": target_timestamps[train_end:val_end],
            "mask": target_mask[train_end:val_end],
            "operational": target_operational[train_end:val_end],
        },
        "test": {
            "x": x[val_end:],
            "y_load": y_load[val_end:],
            "y_pv": y_pv[val_end:],
            "timestamps": target_timestamps[val_end:],
            "mask": target_mask[val_end:],
            "operational": target_operational[val_end:],
        }
    }
    return splits

def main():
    parser = argparse.ArgumentParser(description="Train EARNe baselines")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config yaml file"
    )
    parser.add_argument(
        "--data_root", type=str, default=None, help="Override data_root path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save trained model"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to the baselines folder
        config_path = Path(__file__).resolve().parent / args.config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = load_yaml(config_path)
    print(f"Loaded config: {cfg}")

    data_root = args.data_root if args.data_root is not None else cfg["data_root"]
    seq_len = cfg["seq_len"]
    train_split = cfg.get("train_split", 0.6)
    val_split = cfg.get("val_split", 0.8)
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])

    # Load splits
    splits = load_data(data_root, seq_len, train_split, val_split)
    train_data = splits["train"]
    val_data = splits["val"]

    model_type = cfg["model"]
    print(f"Initializing baseline model: {model_type}")

    if model_type == "persistence":
        model = PersistenceBaseline(quantiles=quantiles)
        model.fit(train_data["x"], train_data["y_load"], train_data["y_pv"])
    elif model_type == "historical_mean":
        model = HistoricalMeanBaseline(quantiles=quantiles)
        model.fit(
            train_data["x"],
            train_data["y_load"],
            train_data["y_pv"],
            train_data["timestamps"]
        )
    elif model_type == "mlp":
        model = MLPBaseline(
            seq_len=seq_len,
            hidden_dim=cfg.get("hidden_dim", 128),
            quantiles=quantiles,
            n_epochs=cfg.get("n_epochs", 50),
            lr=cfg.get("lr", 0.001),
            batch_size=cfg.get("batch_size", 256),
            device="auto"
        )
        model.fit(
            train_data["x"],
            train_data["y_load"],
            train_data["y_pv"],
            val_data["x"],
            val_data["y_load"],
            val_data["y_pv"]
        )
    elif model_type == "bilstm":
        model = BiLSTMBaseline(
            seq_len=seq_len,
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            quantiles=quantiles,
            n_epochs=cfg.get("n_epochs", 15),
            lr=cfg.get("lr", 0.001),
            batch_size=cfg.get("batch_size", 4096),
            device="auto"
        )
        model.fit(
            train_data["x"],
            train_data["y_load"],
            train_data["y_pv"],
            val_data["x"],
            val_data["y_load"],
            val_data["y_pv"]
        )
    elif model_type == "cvae":
        model = CVAEBaseline(
            seq_len=seq_len,
            latent_dim=cfg.get("latent_dim", 32),
            hidden_dim=cfg.get("hidden_dim", 64),
            decoder_dim=cfg.get("decoder_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            quantiles=quantiles,
            n_epochs=cfg.get("n_epochs", 20),
            lr=cfg.get("lr", 0.001),
            batch_size=cfg.get("batch_size", 4096),
            solar_weight=cfg.get("solar_weight", 1.0),
            gate_weight=cfg.get("gate_weight", 1.0),
            kl_weight=cfg.get("kl_weight", 0.1),
            daytime_threshold=cfg.get("daytime_threshold", 0.005),
            pv_eps=cfg.get("pv_eps", 0.001),
            device="auto"
        )
        model.fit(
            train_data["x"],
            train_data["y_load"],
            train_data["y_pv"],
            train_data["timestamps"],
            val_data["x"],
            val_data["y_load"],
            val_data["y_pv"],
            val_data["timestamps"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / f"{model_type}_model.pt"
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
