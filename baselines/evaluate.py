"""Evaluation script for EARNe baselines."""

import argparse
import json
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
from custom_graphgym.transform.transform import Transform
from train import load_data, load_yaml

def compute_pinball_loss(preds, targets, quantiles, mask):
    # preds: [S, N, Q]
    # targets: [S, N]
    # mask: [S, N]
    targets = targets.unsqueeze(-1)
    errors = targets - preds  # [S, N, Q]
    q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device)
    loss = torch.max(q * errors, (q - 1) * errors)  # [S, N, Q]
    masked_loss = loss.mean(dim=-1) * mask  # [S, N]
    return masked_loss.sum().item() / (mask.sum().item() + 1e-9)

def main():
    parser = argparse.ArgumentParser(description="Evaluate EARNe baselines")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config yaml file"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to saved model file"
    )
    parser.add_argument(
        "--data_root", type=str, default=None, help="Override data_root path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save evaluation metrics"
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
    test_data = splits["test"]

    model_type = cfg["model"]

    # Locate saved model
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        model_path = Path(args.output_dir) / f"{model_type}_model.pt"

    print(f"Loading model: {model_type} from {model_path}")
    if model_type == "persistence":
        if model_path.exists():
            model = PersistenceBaseline.load(model_path)
        else:
            print("Model path not found. Initializing stateless persistence baseline.")
            model = PersistenceBaseline(quantiles=quantiles)
    elif model_type == "historical_mean":
        if not model_path.exists():
            raise FileNotFoundError(f"Saved historical mean model not found at {model_path}")
        model = HistoricalMeanBaseline.load(model_path)
    elif model_type == "mlp":
        if not model_path.exists():
            raise FileNotFoundError(f"Saved MLP model not found at {model_path}")
        model = MLPBaseline.load(model_path)
    elif model_type == "bilstm":
        if not model_path.exists():
            raise FileNotFoundError(f"Saved BiLSTM model not found at {model_path}")
        model = BiLSTMBaseline.load(model_path)
    elif model_type == "cvae":
        if not model_path.exists():
            raise FileNotFoundError(f"Saved CVAE model not found at {model_path}")
        model = CVAEBaseline.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Generate predictions
    print("Generating predictions on test set...")
    x_test = test_data["x"]
    if model_type in ("historical_mean", "cvae"):
        preds = model.predict(x_test, test_data["timestamps"])
    else:
        preds = model.predict(x_test)

    # ── Load Transform for Denormalization ──
    data_root_path = Path(data_root)
    transform_path = data_root_path / "transform_single.pt"
    if not transform_path.exists():
        # Try relative to the script location
        transform_path = Path(__file__).resolve().parent / data_root / "transform_single.pt"
    if not transform_path.exists():
        raise FileNotFoundError(f"Could not find transform_single.pt in {data_root}")

    print(f"Loading Transform object from: {transform_path}")
    transform_obj = Transform.load(transform_path)

    # Evaluate Pinball Loss (on scaled values)
    mask = test_data["mask"]
    y_load = test_data["y_load"]
    y_pv = test_data["y_pv"]

    pinball_load = compute_pinball_loss(preds["load"], y_load, quantiles, mask)
    pinball_pv = compute_pinball_loss(preds["pv"], y_pv, quantiles, mask)

    # Evaluate MAE (on denormalized values)
    assert 0.5 in quantiles, "median (0.5) must be in quantiles list for MAE computation!"
    median_idx = quantiles.index(0.5)

    pred_load_median = preds["load"][:, :, median_idx]
    pred_pv_median = preds["pv"][:, :, median_idx]

    inv_pred_load = transform_obj.inverse_transform("load", pred_load_median)
    inv_pred_pv = transform_obj.inverse_transform("pv", pred_pv_median)
    inv_true_load = transform_obj.inverse_transform("load", y_load)
    inv_true_pv = transform_obj.inverse_transform("pv", y_pv)

    err_load = torch.abs(inv_pred_load - inv_true_load)
    err_pv = torch.abs(inv_pred_pv - inv_true_pv)

    mae_load = (err_load * mask).sum().item() / (mask.sum().item() + 1e-9)
    mae_pv = (err_pv * mask).sum().item() / (mask.sum().item() + 1e-9)

    metrics = {
        "model": model_type,
        "pinball_loss_load": float(round(pinball_load, 5)),
        "pinball_loss_pv": float(round(pinball_pv, 5)),
        "mae_load_w": float(round(mae_load, 3)),
        "mae_pv_w": float(round(mae_pv, 3)),
    }

    print("\nEvaluation Metrics:")
    print(f"  Model:              {metrics['model']}")
    print(f"  Pinball Loss Load:  {metrics['pinball_loss_load']:.5f}")
    print(f"  Pinball Loss PV:    {metrics['pinball_loss_pv']:.5f}")
    print(f"  MAE Load (W):       {metrics['mae_load_w']:.3f}")
    print(f"  MAE PV (W):         {metrics['mae_pv_w']:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = output_dir / f"{model_type}_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_save_path}")

if __name__ == "__main__":
    main()
