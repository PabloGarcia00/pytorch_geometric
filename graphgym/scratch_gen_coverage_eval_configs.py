"""One-off script: generates single-read configs for all 7 models (GNN, MLP,
LSTM, CVAE, KNN, Linear, SVR) contrasting optimal vs sub-optimal household
coverage on the existing 15min gold-layer dataset, mirroring the naming
convention already established for st_sgc_caps (st_sgc_caps_single_full_
optimal/suboptimal.yaml):
  - optimal    = earne_data.require_full_span: true  (drops nodes that
    onboarded late / offboarded early -- cleaner per-household history)
  - suboptimal = earne_data.require_full_span: false (native observation
    window, all nodes included regardless of partial coverage)

require_full_span is only read in EARNeGraphDataset._get_node_mask(), which
runs in __init__ *after* the cached master bundle is loaded -- it never
affects what process() writes to disk. So both variants safely reuse the
exact same already-built dataset cache (datasets/earne_csi /
datasets/earne_cvae_csi for CVAE) with zero new dataset build required.
"""

import copy
from pathlib import Path

import yaml

OUT_DIR = Path("configs/pyg")

TEMPLATES = {
    "gnn": "configs/pyg/earne_learned_corr_single_csi.yaml",
    "mlp": "configs/pyg/baseline_mlp_single_csi.yaml",
    "lstm": "configs/pyg/baseline_lstm_single_csi.yaml",
    "cvae": "configs/pyg/baseline_cvae_single_csi.yaml",
    "knn": "configs/pyg/baseline_knn_single_csi.yaml",
    "linear": "configs/pyg/baseline_linear_single_csi.yaml",
    "svr": "configs/pyg/baseline_svr_single_csi.yaml",
}

COVERAGE = {"optimal": True, "suboptimal": False}

written = []
for model, template_path in TEMPLATES.items():
    base = yaml.safe_load(Path(template_path).read_text())
    for cov, require_full_span in COVERAGE.items():
        cfg = copy.deepcopy(base)
        name = f"coverage_eval_{model}_{cov}"

        cfg.setdefault("earne_data", {})["require_full_span"] = require_full_span
        cfg.setdefault("dataset", {})["name"] = f"earne_{name}"
        cfg.setdefault("train", {}).setdefault("wandb", {})["project"] = "earne-coverage-eval"

        if model == "knn" and cov == "suboptimal":
            # suboptimal (require_full_span=False) keeps ~112 households vs
            # ~62 for optimal -- confirmed via a real OOM crash that the
            # GPU-resident neighbor bank doesn't scale to the larger node
            # count either (same underlying issue as the 5min-resolution
            # case, see project_knn_resolution_scaling memory). optimal
            # already works fine on GPU, so only override here.
            cfg.setdefault("baseline", {})["knn_cache_device"] = "cpu"

        if model == "linear":
            # Policy: canonical Linear/SVR baselines always use 'current'
            # feature mode, never 'window' -- see scratch_gen_resolution_
            # eval_configs.py for the full rationale (window-mode
            # destabilizes under nonzero physics_weight and underperforms
            # current-mode generally).
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
