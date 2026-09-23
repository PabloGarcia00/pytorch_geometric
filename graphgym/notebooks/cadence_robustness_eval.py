"""
cadence_robustness_eval.py

Phase 2 of ~/.claude/plans/squishy-wobbling-bunny.md ("Robustness Test Plan
for Input Cadence Mismatch at Inference Time"): inference-only comparison of
a 5-min-trained checkpoint's accuracy under its native clean 288x5-min window
("control") vs. the mixed-cadence window (283x5-min + 5x1-min, "mixed") --
using onemin_stream_adapter.synthesize_oneminute() as the 1-min data source,
since the real 1-minute smart-meter source (Phase 0) is still unlocated on
this box. No retraining -- the checkpoint from run_5min_dualmask.sh is used
as-is, exactly as the plan's design principles require.

This does NOT reimplement custom_graphgym/eval/streaming_harness.py's
WindowState -- it builds each mixed sample directly from the existing
test-set Data objects (see build_mixed_sample() docstring for why: every
non-consumption channel in a training sample -- generation, weather,
operational -- has no genuine 1-minute analog either, so "streaming" one
tick at a time and re-deriving them would just reinvent the same
hold-constant choice this makes explicitly).

Usage:
    python notebooks/cadence_robustness_eval.py --cfg configs/pyg/res_eval_lstm_5min_dualmask.yaml \
        --ckpt results/res_eval_lstm_5min_dualmask/0/ckpt/epoch=16-step=117283.ckpt \
        --arch lstm --n-samples 100

Writes/merges into notebooks/cadence_robustness_long.csv (schema:
arch, condition, mitigation, device, target, metric, value) -- kept
separate from metrics_long.csv per the plan's explicit "never merge into
the main pipeline" instruction.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import custom_graphgym  # noqa — registers loaders/losses/metrics
import numpy as np
import pandas as pd
import torch
from custom_graphgym.loader.graph_dataset import _encode_temporal
from custom_graphgym.metric.regression import DisaggregationMetrics
from custom_graphgym.target_utils import active_targets

from onemin_stream_adapter import synthesize_oneminute

from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.loader import create_dataset
from torch_geometric.graphgym.model_builder import create_model

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

FINE_LEN = 5  # matches streaming_harness.WindowState.FINE_LEN


def _load_state_dict(ckpt_path: Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if model_keys[0].startswith("model.") and not ckpt_keys[0].startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif ckpt_keys[0].startswith("model.") and not model_keys[0].startswith("model."):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


CACHE_FIT_MODEL_TYPES = {"baseline_knn"}  # matches calculate_metrics.py — no checkpoint


def load_model_and_dataset(cfg_path: str, ckpt_path: str | None):
    cfg.set_new_allowed(True)
    load_cfg(cfg, argparse.Namespace(cfg_file=cfg_path, opts=[]))
    # Lets EARNeGraphDataset's results-dir-first transform lookup find a
    # results/<run>/transform_*.pt copy before falling back to datasets/ --
    # see graph_dataset.py and main.py's _copy_transform_to_results().
    cfg.out_dir = str(Path(cfg_path).parent)
    cfg.num_workers = 0
    cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.baseline.knn_cache_device = cfg.accelerator
    device = torch.device(cfg.accelerator)

    dataset = create_dataset()
    model = create_model()
    if cfg.model.type in CACHE_FIT_MODEL_TYPES:
        train_idx, _, _ = dataset.get_split_indices()
        from torch_geometric.loader import DataLoader as PyGDataLoader
        train_loader = PyGDataLoader(
            dataset[torch.tensor(train_idx, dtype=torch.long)],
            batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, pin_memory=True,
        )
        model.model.fit_cache(train_loader)
    else:
        _load_state_dict(Path(ckpt_path), model)
    model = model.to(device).eval()
    return dataset, model, device


def build_mixed_sample(dataset, idx: int):
    """
    Build (control, mixed) Data objects for target index `idx` (tgt =
    idx + seq_len). Both predict the identical target (y_load/y_pv/mask/
    y_net_demand at tgt) so they're directly comparable.

    `mixed` differs from `control` only in the CONSUMPTION channel's most
    recent 5 minutes: instead of one 5-min-aggregate slot (position 287,
    the same value control uses), it holds 5 discrete 1-minute readings
    synthesized via synthesize_oneminute() from the two raw 5-min knots
    bracketing that slot -- recovered via dataset.transform_obj's own
    inverse_transform, so no separate raw data path is needed. This drops
    the oldest 4 5-min slots (283-286) to keep a fixed 288-length window,
    matching streaming_harness.WindowState's 283+5 split (~23.6h of
    context instead of a clean 24h -- see that module's docstring).

    Every other channel (generation/PV, weather, operational) has no
    genuine 1-minute analog in this dataset -- PV is natively 5-min (the
    inverter's own ceiling), weather/operational have no finer source --
    so they're held constant at their last-slot value across the new 5
    fine positions, exactly as streaming_harness.py's own design notes
    say to. Calendar features (`temporal`) DO have a well-defined value at
    any real timestamp (pure sin/cos-of-hour/day/month, no data lookup),
    so those are recomputed properly via _encode_temporal rather than
    repeated.
    """
    control = dataset.get(idx)
    tgt = idx + dataset.seq_len
    timestamps = pd.to_datetime(dataset.timestamps)

    t_prev2 = timestamps[tgt - 2] if tgt - 2 >= 0 else timestamps[tgt - 1]
    t_prev1 = timestamps[tgt - 1]

    raw_prev2 = dataset.transform_obj.inverse_transform(
        "consumption", dataset.consumption_scaled[tgt - 2]
    )
    raw_prev1 = dataset.transform_obj.inverse_transform(
        "consumption", dataset.consumption_scaled[tgt - 1]
    )

    n = dataset.num_nodes
    knot_df = pd.DataFrame(
        {
            "timestamp": [t_prev2] * n + [t_prev1] * n,
            "user_id": list(range(n)) * 2,
            "consumption_w": torch.cat([raw_prev2, raw_prev1]).numpy(),
        }
    )
    one_min = synthesize_oneminute(knot_df, value_col="consumption_w")
    # 5 real one-minute readings strictly after t_prev2, up to and
    # including t_prev1 -- the FINE_LEN window "now" would actually see.
    fine_raw = torch.zeros(FINE_LEN, n)
    fine_times = []
    for uid, group in one_min.groupby("user_id", sort=True):
        g = group.sort_values("timestamp")
        tail = g.iloc[-FINE_LEN:]
        fine_raw[:, uid] = torch.tensor(tail["consumption_w"].to_numpy(), dtype=torch.float32)
        if uid == 0:
            fine_times = list(tail["timestamp"])

    fine_scaled = dataset.transform_obj.transform("consumption", fine_raw)  # [5, N]

    mixed = control.clone()
    coarse_x = control.x[:, : -FINE_LEN, :]  # [N, 283, C] — unchanged history
    fine_generation = control.x[:, -1:, 1:2].expand(-1, FINE_LEN, -1) if control.x.shape[-1] > 1 else None
    fine_consumption = fine_scaled.t().unsqueeze(-1)  # [N, 5, 1]
    if fine_generation is not None:
        fine_x = torch.cat([fine_consumption, fine_generation], dim=-1)
    else:
        fine_x = fine_consumption
    mixed.x = torch.cat([coarse_x, fine_x], dim=1)

    if control.temporal is not None:
        fine_temporal = _encode_temporal(pd.DatetimeIndex(fine_times))
        mixed.temporal = torch.cat([control.temporal[:-FINE_LEN], fine_temporal], dim=0)

    if control.operational is not None:
        fine_operational = control.operational[:, -1:, :].expand(-1, FINE_LEN, -1)
        mixed.operational = torch.cat(
            [control.operational[:, :-FINE_LEN, :], fine_operational], dim=1
        )

    if getattr(control, "weather", None) is not None:
        fine_weather = control.weather[:, -1:, :].expand(-1, FINE_LEN, -1)
        mixed.weather = torch.cat([control.weather[:, :-FINE_LEN, :], fine_weather], dim=1)

    return control, mixed


def run(cfg_path: str, ckpt_path: str, arch: str, n_samples: int):
    dataset, model, device = load_model_and_dataset(cfg_path, ckpt_path)
    _, _, test_idx = dataset.get_split_indices()
    sample_idx = np.linspace(0, len(test_idx) - 1, num=min(n_samples, len(test_idx)), dtype=int)
    sample_idx = sorted({test_idx[i] for i in sample_idx})

    targets = active_targets()
    n_q = cfg.model.n_quantiles
    t = dataset.transform_obj

    results = {"control_5min": ([], []), "mixed_cadence": ([], [])}
    trues = []

    with torch.no_grad():
        for idx in sample_idx:
            control, mixed = build_mixed_sample(dataset, idx)
            trues.append(control.y_load.clone())  # placeholder, replaced below
            for cond, data in (("control_5min", control), ("mixed_cadence", mixed)):
                batch = Batch.from_data_list([data]).to(device)
                pred, true = model(batch)
                results[cond][0].append(true.cpu())
                results[cond][1].append(pred.cpu())

    rows = []
    for cond, (true_chunks, pred_chunks) in results.items():
        true = torch.stack(true_chunks, dim=-1)  # [N, 4, T]
        pred = torch.stack(pred_chunks, dim=-1)  # [N, n_q*len(targets), T]

        pred_denorm = torch.cat(
            [
                t.inverse_transform(name, pred[:, i * n_q : (i + 1) * n_q])
                for i, name in enumerate(targets)
            ],
            dim=1,
        )
        true_denorm = torch.cat(
            [
                t.inverse_transform("load", true[:, [0]]),
                t.inverse_transform("pv", true[:, [1]]),
                true[:, [2]],
                t.inverse_transform("net_demand", true[:, [3]]),
            ],
            dim=1,
        )
        metrics = DisaggregationMetrics.all(
            true_denorm.numpy(), pred_denorm.numpy(),
            targets=targets, n_quantiles=n_q, q50_idx=cfg.model.quantiles.index(0.5),
        )
        for target, target_metrics in metrics.items():
            for metric, value in target_metrics.items():
                rows.append(
                    {
                        "arch": arch,
                        "condition": cond,
                        "mitigation": "none",
                        "device": device.type,
                        "target": target,
                        "metric": metric,
                        "value": value,
                    }
                )

    out = pd.DataFrame(rows)
    out_path = Path("notebooks/cadence_robustness_long.csv")
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["arch"] != arch]
        out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(out)} rows total, {len(rows)} from arch={arch})")
    print(f"n_samples used: {len(sample_idx)} / {len(test_idx)} test-set windows")

    piv = out[out["arch"] == arch].pivot_table(index="condition", columns="metric", values="value")
    print(piv[["rmse", "nrmse", "mae", "nmae", "r2", "coverage", "sharpness"]].round(4)
          if "rmse" in piv.columns else piv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    run(args.cfg, args.ckpt, args.arch, args.n_samples)
