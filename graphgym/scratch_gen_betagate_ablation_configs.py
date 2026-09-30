"""One-off script: generates the 3 encoders x 2 heads ablation testing
whether EARNeBetaGateHead's Gaussian-load + zero-inflated-Beta-PV output
(vs. earne_quantile's direct pinball-loss quantile regression) is what
drives good PV disaggregation, independent of encoder architecture.

Every run shares the same data slice (15-min/native cadence, dual-read,
weather off, physics-masked, full-span/"optimal") -- see
configs/pyg/earne_learned_corr_dual_physmask.yaml / baseline_lstm_dual_
physmask.yaml / baseline_mlp_dual_physmask.yaml, the three templates this
mirrors -- and only head_name/loss_fun differ within each encoder pair.

Dataset override, v2: the three templates above all point at datasets/
earne_dual_physmask, which normalizes pv via ihs (unbounded) -- fine for
earne_quantile's direct regression, but wrong for EARNeBetaGateHead's Beta
head, whose support is (0,1). cvae_loss.py's Beta NLL has always paired
with a *separate* minmax-normalized dataset for exactly this reason (see
configs/pyg/baseline_cvae_dual_physmask.yaml -> datasets/
earne_cvae_dual_physmask) -- v1 of this script missed that and pointed all
6 configs at the ihs dataset instead, which clamps every target into a
degenerate band near the clamp boundary (pv_clamped = y_pv.clamp(1e-3,
1-1e-3)) and produces uniformly bad, encoder-independent MAE that looks
like a finding but is actually a preprocessing bug. Both heads now train
on the same minmax dataset (baseline_cvae_dual_physmask.yaml's earne_data
block otherwise matches these templates exactly: seq_len 96, dual_read,
weather off, require_full_span, mask_physics_impossible), so the ablation
isolates head_name/loss_fun as originally intended for BOTH arms, not just
one.

Run names are prefixed "betagate_" so notebooks/calculate_metrics.py's new
_FLAT_SWEEP_GLOBS["betagate_*"] entry picks them up as one sweep.
"""

import copy
from pathlib import Path

import yaml

OUT_DIR = Path("configs/pyg")

TEMPLATES = {
    "gnn": "configs/pyg/earne_learned_corr_dual_physmask.yaml",
    "lstm": "configs/pyg/baseline_lstm_dual_physmask.yaml",
    "mlp": "configs/pyg/baseline_mlp_dual_physmask.yaml",
}

HEADS = {
    "quantile": {"head_name": "earne_quantile", "loss_fun": "earne_loss"},
    "betagate": {"head_name": "earne_beta_gate", "loss_fun": "earne_beta_gate_loss"},
}

written = []
for encoder, template_path in TEMPLATES.items():
    base = yaml.safe_load(Path(template_path).read_text())
    for head, overrides in HEADS.items():
        cfg = copy.deepcopy(base)
        name = f"betagate_{encoder}_{head}"

        cfg["model"]["head_name"] = overrides["head_name"]
        cfg["model"]["loss_fun"] = overrides["loss_fun"]
        cfg.setdefault("dataset", {})["name"] = f"earne_{name}"
        cfg["dataset"]["dir"] = "datasets/earne_cvae_dual_physmask"
        cfg["earne_data"]["processed_root"] = "datasets/earne_cvae_dual_physmask"
        cfg["earne_data"]["energy_norm_mode"]["pv"] = "minmax"
        cfg.setdefault("train", {}).setdefault("wandb", {})["project"] = "earne-beta-gate-ablation"

        out_path = OUT_DIR / f"{name}.yaml"
        out_path.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
        written.append(str(out_path))

print(f"wrote {len(written)} configs")
for p in written:
    print(" ", p)
