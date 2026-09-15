"""One-off script: generates single-read configs for all 7 models (GNN, MLP,
LSTM, CVAE, KNN, Linear, SVR) against 3 new-resolution gold-layer files
(5min, 30min, 60min), evaluating how PV/load disaggregation quality changes
with sampling interval.

Templates cloned as-is (architecture/hyperparameters untouched) from the
existing single-read weather-off baselines (baseline_*_single_csi.yaml,
earne_learned_corr_single_csi.yaml). Only these keys are overridden:
  - earne_data.gold_data -> the resolution-specific parquet
  - model.seq_len -> scaled to preserve the same 24h lookback window as the
    existing 15min/seq_len=96 baseline (288 @ 5min, 48 @ 30min, 24 @ 60min)
  - dataset.dir / earne_data.processed_root -> a new, resolution-specific
    cache dir, shared across the 6 non-CVAE models (dataset content depends
    only on dual_read/weather_mode/seq_len/gold_data/filters, never on
    model architecture -- same convention the existing datasets/earne_csi
    cache already uses); CVAE keeps its own separate cache dir, mirroring
    the existing datasets/earne_cvae_csi split.
  - dataset.name -> cosmetic
  - train.wandb.project -> dedicated project so this batch is visually
    separate in the wandb dashboard (matches the earne-tstr-trts precedent)
"""

import copy
from pathlib import Path

import yaml

OUT_DIR = Path("configs/pyg")
GOLD_DIR = Path("/home/sagemaker-user/exploratory-data-analysis/output/gold_layer")

TEMPLATES = {
    "gnn": "configs/pyg/earne_learned_corr_single_csi.yaml",
    "mlp": "configs/pyg/baseline_mlp_single_csi.yaml",
    "lstm": "configs/pyg/baseline_lstm_single_csi.yaml",
    "cvae": "configs/pyg/baseline_cvae_single_csi.yaml",
    "knn": "configs/pyg/baseline_knn_single_csi.yaml",
    "linear": "configs/pyg/baseline_linear_single_csi.yaml",
    "svr": "configs/pyg/baseline_svr_single_csi.yaml",
}

# seq_len scaled to preserve the same 24h lookback as 15min/seq_len=96
RESOLUTIONS = {
    "5min": {"gold_data": GOLD_DIR / "fleet_gold_layer_5min.parquet", "seq_len": 288},
    "30min": {"gold_data": GOLD_DIR / "fleet_gold_layer_30min.parquet", "seq_len": 48},
    "60min": {"gold_data": GOLD_DIR / "fleet_gold_layer_60min.parquet", "seq_len": 24},
}

written = []
for model, template_path in TEMPLATES.items():
    base = yaml.safe_load(Path(template_path).read_text())
    for res, overrides in RESOLUTIONS.items():
        cfg = copy.deepcopy(base)
        name = f"res_eval_{model}_{res}"

        cache_dir = f"datasets/earne_cvae_csi_{res}" if model == "cvae" else f"datasets/earne_csi_{res}"

        cfg.setdefault("earne_data", {})["gold_data"] = str(overrides["gold_data"])
        cfg["earne_data"]["processed_root"] = cache_dir
        cfg.setdefault("dataset", {})["dir"] = cache_dir
        cfg["dataset"]["name"] = f"earne_{name}"
        cfg.setdefault("model", {})["seq_len"] = overrides["seq_len"]
        cfg.setdefault("train", {}).setdefault("wandb", {})["project"] = "earne-resolution-eval"

        if model == "knn" and res == "5min":
            # 5min's ~3x more training rows x 3x wider window (seq_len=288
            # vs 96) blows well past the profiled 9.69GB GPU bank estimate
            # in config/baseline.py -- confirmed via smoke test: OOM on a
            # 22GB card. Fall back to the documented cpu escape hatch for
            # this one config; 30/60min have fewer rows than 15min (which
            # already works fine on GPU) so they're left as-is.
            cfg.setdefault("baseline", {})["knn_cache_device"] = "cpu"

        if model == "linear":
            # Policy: canonical Linear/SVR baselines always use 'current'
            # feature mode, never 'window' -- window-mode's much wider
            # flattened-history input destabilizes training under a
            # nonzero physics_weight (confirmed: window-mode Linear at
            # pw=0.3 diverged to billions of W MAE) and underperforms
            # current-mode even at pw=0.0. min_delta is needed because
            # current-mode's far-fewer-parameter fit plateaus almost
            # immediately and would otherwise never trigger early stopping.
            cfg.setdefault("baseline", {})["linear_feature_mode"] = "current"
            cfg.setdefault("train", {}).setdefault("early_stopping", {})["min_delta"] = 0.001
        elif model == "svr":
            cfg.setdefault("baseline", {})["svr_feature_mode"] = "current"
            cfg["baseline"]["svr_rff_dim"] = 32
            cfg.setdefault("train", {}).setdefault("early_stopping", {})["min_delta"] = 0.001
            cfg.setdefault("optim", {})["weight_decay"] = 1.0e-05

        out_path = OUT_DIR / f"{name}.yaml"
        out_path.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
        written.append(str(out_path))

print(f"wrote {len(written)} configs")
for p in written:
    print(" ", p)
