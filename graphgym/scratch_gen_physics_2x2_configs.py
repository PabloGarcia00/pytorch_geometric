"""One-off script: completes the physics_weight x mask_physics_impossible 2x2
for all 7 models. Existing coverage before this script:
  (mask=True,  pw=0.0) -- configs/baseline_<model>_dual_physmask_grid_sweep_physics_weight/*-pw=0.0.yaml
  (mask=True,  pw=0.3) -- .../*-pw=0.3.yaml
  (mask=False, pw=0.0) -- configs/pyg/baseline_<model>_dual[_nomask].yaml / earne_learned_corr_dual.yaml
  (mask=False, pw=0.3) -- MISSING (this script)

Clones each model's existing (mask=False, pw=0.0) template as-is and
overrides only train.physics_weight -> 0.3. mask_physics_impossible only
affects EARNeGraphDataset.process() (it's a data-level mask baked into the
cached tensors, unlike require_full_span's post-hoc node filtering) --
confirmed the physmask/nomask variants already point at *different* cache
dirs (datasets/earne_dual_physmask vs datasets/earne_dual_csi /
earne_cvae_dual_nomask_csi). Since this new contrast keeps
mask_physics_impossible=False, it reuses the exact same already-built
dataset.dir/processed_root as the existing nomask baseline -- zero new
dataset builds needed.
"""

import copy
from pathlib import Path

import yaml

OUT_DIR = Path("configs/pyg")

TEMPLATES = {
    "gnn": "configs/pyg/earne_learned_corr_dual.yaml",
    "mlp": "configs/pyg/baseline_mlp_dual_nomask.yaml",
    "lstm": "configs/pyg/baseline_lstm_dual_nomask.yaml",
    "cvae": "configs/pyg/baseline_cvae_dual_nomask.yaml",
    "knn": "configs/pyg/baseline_knn_dual.yaml",
    "linear": "configs/pyg/baseline_linear_dual.yaml",
    "svr": "configs/pyg/baseline_svr_dual.yaml",
}

written = []
for model, template_path in TEMPLATES.items():
    base = yaml.safe_load(Path(template_path).read_text())
    cfg = copy.deepcopy(base)

    cfg.setdefault("train", {})["physics_weight"] = 0.3
    cfg["train"].setdefault("wandb", {})["project"] = "earne-physics-2x2"

    if model == "linear":
        # Policy: canonical Linear/SVR baselines always use 'current' feature
        # mode, never 'window'. Concretely confirmed necessary here: window-
        # mode Linear at pw=0.3 diverged to billions of W test MAE (val_loss
        # looked completely normal throughout -- the same "loss descends
        # smoothly while MAE explodes" signature as the original pre-fix
        # physics-weight-dominance bug, just far more extreme). The existing
        # x0.1 physics_weight scale-down for canonical models was only ever
        # validated against single-read/current-mode; it doesn't stabilize
        # dual-read/window-mode's much higher-dimensional input.
        cfg.setdefault("baseline", {})["linear_feature_mode"] = "current"
        cfg.setdefault("train", {}).setdefault("early_stopping", {})["min_delta"] = 0.001
    elif model == "svr":
        cfg.setdefault("baseline", {})["svr_feature_mode"] = "current"
        cfg["baseline"]["svr_rff_dim"] = 32
        cfg.setdefault("train", {}).setdefault("early_stopping", {})["min_delta"] = 0.001
        cfg.setdefault("optim", {})["weight_decay"] = 1.0e-05

    template_name = Path(template_path).stem
    out_path = OUT_DIR / f"{template_name}_pw03.yaml"
    out_path.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    written.append(str(out_path))

print(f"wrote {len(written)} configs")
for p in written:
    print(" ", p)
