import os
import sys
from pathlib import Path
import argparse
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'graphgym'))

from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.loader import create_loader
import custom_graphgym  # noqa

# 15-min resolution: 96 steps per day, 672 per week
STEPS_PER_DAY = 96
STEPS_PER_WEEK = STEPS_PER_DAY * 7

# Weather feature encoding: last 4 columns of the weather tensor are
# [cos(day-of-week), sin(day-of-week), cos(time-of-day), sin(time-of-day)]
# cos=1, sin=0 => Monday at 00:00
_W_COS_TOD = -1   # cos(time-of-day)
_W_SIN_TOD = -2   # sin(time-of-day)
_W_COS_DOW = -3   # cos(day-of-week)
_W_SIN_DOW = -4   # sin(day-of-week)


def unwrap_dataset(dataset):
    """Return the underlying dataset, unwrapping a Subset wrapper if present."""
    return dataset.dataset if hasattr(dataset, 'dataset') else dataset


def init_style():
    plt.style.use("default")
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        # tight bounding box on save; tight_layout() handles internal spacing
        "savefig.bbox": "tight",
    })


# --- Experiment setup (split into focused helpers) ---

def _find_config(exp_path: Path) -> Path:
    """Locate config.yaml starting from exp_path, then parent, then recursive."""
    for candidate in [exp_path / "config.yaml", exp_path.parent / "config.yaml"]:
        if candidate.exists():
            return candidate
    configs = list(exp_path.glob("**/config.yaml"))
    if configs:
        return configs[0]
    raise FileNotFoundError(f"Could not find config.yaml in {exp_path}")


def _find_latest_checkpoint(exp_path: Path) -> Path:
    """Return the most recently modified .ckpt file under exp_path."""
    ckpt_dir = exp_path / "ckpt"
    ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else list(exp_path.glob("**/ckpt/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {exp_path}")
    return sorted(ckpts, key=os.path.getmtime)[-1]


def _load_model(latest_ckpt: Path, device: torch.device):
    """Instantiate model from cfg and load checkpoint weights."""
    model = create_model()
    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))

    # Reconcile 'model.' prefix between GraphGymModule and bare model keys
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if model_keys[0].startswith('model.') and not ckpt_keys[0].startswith('model.'):
        state_dict = {f'model.{k}': v for k, v in state_dict.items()}
    elif ckpt_keys[0].startswith('model.') and not model_keys[0].startswith('model.'):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def setup_experiment(exp_dir):
    """Find config + checkpoint, load cfg, return (model, device)."""
    exp_path = Path(exp_dir)
    config_path = _find_config(exp_path)
    latest_ckpt = _find_latest_checkpoint(exp_path)

    print(f"--- Experiment Setup ---")
    print(f"Directory: {exp_dir}")
    print(f"Config:    {config_path}")
    print(f"Model:     {latest_ckpt}")

    cfg.set_new_allowed(True)
    load_cfg(cfg, argparse.Namespace(cfg_file=str(config_path), opts=[]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = _load_model(latest_ckpt, device)
    return model, device


def active_diverse_selection(dataset, node_idx, candidates, num_steps, k=2, threshold=1e-5):
    """Greedily select k diverse, active windows from candidates.

    'Active' means both load and PV have std > threshold.
    Diversity is maximised by the farthest-first (maximin) heuristic.
    """
    full_ds = unwrap_dataset(dataset)
    print(f"Searching for {k} active/diverse windows among {len(candidates)} candidates...")

    active_cand, active_prof = [], []
    seq_len = cfg.model.seq_len

    for idx in candidates:
        if idx + seq_len + num_steps > full_ds.load_scaled.shape[0]:
            continue

        # Prediction target window starts at idx + seq_len
        l_prof = full_ds.load_scaled[idx + seq_len: idx + seq_len + num_steps, node_idx].cpu().numpy()
        p_prof = full_ds.pv_scaled[idx + seq_len: idx + seq_len + num_steps, node_idx].cpu().numpy()

        if np.std(l_prof) > threshold and np.std(p_prof) > threshold:
            active_cand.append(idx)
            active_prof.append(np.concatenate([l_prof, p_prof]))

    if not active_cand:
        print(f"Warning: No candidates met threshold {threshold}. Returning first {k} available.")
        return candidates[:k]

    # Seed with highest-variance profile, then greedily add the most distant
    variances = [np.var(p) for p in active_prof]
    best = int(np.argmax(variances))
    selected_idx = [active_cand[best]]
    selected_prof = [active_prof[best]]

    while len(selected_idx) < k and len(active_cand) > len(selected_idx):
        max_min_dist, best_i = -1, -1
        for i, p in enumerate(active_prof):
            if active_cand[i] in selected_idx:
                continue
            min_dist = min(np.linalg.norm(p - sp) for sp in selected_prof)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_i = i

        if best_i == -1:
            break
        selected_idx.append(active_cand[best_i])
        selected_prof.append(active_prof[best_i])

    print(f"Successfully selected diverse indices: {selected_idx}")
    return selected_idx


def pprint_results(res, label=""):
    """Print full values for every array in the results dict (kW, 4 d.p.)."""
    import pprint as _pp
    header = f"--- Results{' ' + label if label else ''} ---"
    print(header)
    for key, val in res.items():
        if isinstance(val, np.ndarray):
            # Round for readability; tolist() gives plain Python floats for pprint
            print(f"  {key} {val.shape}:")
            _pp.pprint(np.round(val, 4).tolist(), indent=4)
        else:
            print(f"  {key}: {val}")
    print("-" * len(header))


def collect_results(model, dataset, node_idx, start_idx, num_steps, device, scaler):
    """Run model inference and denormalise predictions for one window.

    Args:
        scaler: pre-loaded sklearn scaler (pass once, reuse across calls).
    """
    full_ds = unwrap_dataset(dataset)
    actual_load, actual_pv, pred_load, pred_pv = [], [], [], []
    n_q = cfg.model.n_quantiles  # output layout: [load_q0..qN, pv_q0..qN]

    print(f"Collecting predictions for index {start_idx}...")
    with torch.no_grad():
        for i in range(num_steps):
            batch = full_ds.get(start_idx + i).to(device)
            pred, _ = model(batch)
            actual_load.append(batch.y_load[node_idx].item())
            actual_pv.append(batch.y_pv[node_idx].item())
            pred_load.append(pred[node_idx, :n_q].cpu().numpy())
            pred_pv.append(pred[node_idx, n_q:2 * n_q].cpu().numpy())

    res = {
        'actual_load': np.array(actual_load),
        'actual_pv': np.array(actual_pv),
        'pred_load': np.array(pred_load),
        'pred_pv': np.array(pred_pv),
    }

    # Denormalise (reshape to (-1,1) for scaler) then convert W -> kW
    for key, v in res.items():
        res[key] = scaler.inverse_transform(v.reshape(-1, 1)).reshape(v.shape) / 1000.0

    # Timestamps for subplot titles
    seq_len = cfg.model.seq_len
    if hasattr(full_ds, 'timestamps') and full_ds.timestamps is not None:
        res['title_start'] = str(full_ds.timestamps[start_idx + seq_len])[:10]
        res['title_end'] = str(full_ds.timestamps[start_idx + seq_len + num_steps - 1])[:10]
    else:
        res['title_start'] = f"Index {start_idx + seq_len}"
        res['title_end'] = f"Index {start_idx + seq_len + num_steps}"

    pprint_results(res, label=f"start_idx={start_idx}, node={node_idx}")
    return res


def plot_multi(results_list, out_path, is_weekly):
    """Generate a multi-panel figure: 1×N horizontal (weekly) or N×1 vertical (daily)."""
    num_samples = len(results_list)

    if is_weekly:
        # Wide figure: one panel per week, side by side
        fig, axes = plt.subplots(1, num_samples, figsize=(14.0, 3.5))
    else:
        # Narrow figure: one panel per day, stacked
        fig, axes = plt.subplots(num_samples, 1, figsize=(5, 4.5))

    if num_samples == 1:
        axes = [axes]

    load_color = '#4A90E2'
    pv_color = '#7ED321'

    for i, res in enumerate(results_list):
        ax = axes[i]
        time_x = np.arange(len(res['actual_load']))

        # Consumption (Load)
        ax.plot(time_x, res['actual_load'], color=load_color, lw=0.6, ls='-', label='Real Consumption')
        ax.plot(time_x, res['pred_load'][:, 1], color=load_color, lw=0.8, ls='-.', label='Estimated Consumption')
        ax.fill_between(time_x, res['pred_load'][:, 0], res['pred_load'][:, -1],
                        color=load_color, alpha=0.15, label='Consumption 80% CI')

        # Generation (PV)
        ax.plot(time_x, res['actual_pv'], color=pv_color, lw=0.6, ls='-', label='Real PV Generation')
        ax.plot(time_x, res['pred_pv'][:, 1], color=pv_color, lw=0.8, ls='-.', label='Estimated PV Generation')
        ax.fill_between(time_x, res['pred_pv'][:, 0], res['pred_pv'][:, -1],
                        color=pv_color, alpha=0.15, label='PV 80% CI')

        # Axis formatting
        if is_weekly:
            assert len(time_x) == STEPS_PER_WEEK, (
                f"Expected {STEPS_PER_WEEK} steps for weekly plot, got {len(time_x)}"
            )
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']
            ax.set_xticks(np.arange(0, STEPS_PER_WEEK + 1, STEPS_PER_DAY))
            ax.set_xticklabels(day_labels)
            for d in range(1, 7):
                ax.axvline(d * STEPS_PER_DAY, color='k', alpha=0.05, lw=0.5)
        else:
            hour_labels = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00']
            ax.set_xticks(np.linspace(0, STEPS_PER_DAY, 9))
            ax.set_xticklabels(hour_labels)

        ax.set_ylabel('Power (kW)')
        ax.grid(axis='y', ls='--', alpha=0.4)

        # Legend on the first panel only, with explicit handle order:
        # Real Consumption | Estimated Consumption | Real PV | Estimated PV | Load 80% CI | PV 80% CI
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            order = [
                'Real Consumption', 'Estimated Consumption',
                'Real PV Generation', 'Estimated PV Generation',
                'Consumption 80% CI', 'PV 80% CI',
            ]
            label_to_handle = dict(zip(labels, handles))
            sorted_handles = [label_to_handle[l] for l in order if l in label_to_handle]
            sorted_labels  = [l for l in order if l in label_to_handle]
            anchor = (1.1, 1.45) if is_weekly else (0.5, 1.45)
            ax.legend(sorted_handles, sorted_labels,
                      loc='upper center', bbox_to_anchor=anchor, ncol=3, frameon=False)

    # tight_layout adjusts subplot spacing; savefig.bbox='tight' trims the output canvas
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"{'Weekly (1x2)' if is_weekly else 'Daily (2x1)'} Plot saved to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True,
                        help='Experiment directory (results/path/to/exp/0/)')
    parser.add_argument('--node', type=int, default=0)
    parser.add_argument('--weekly', action='store_true')
    parser.add_argument('--out', type=str, default='profile_sample.pdf')
    args = parser.parse_args()

    init_style()

    model, device = setup_experiment(args.dir)
    loaders = create_loader()
    dataset = loaders[2].dataset
    full_ds = unwrap_dataset(dataset)
    num_steps = STEPS_PER_WEEK if args.weekly else STEPS_PER_DAY

    # Collect indices whose input window starts at 00:00 (and Monday for weekly)
    if hasattr(dataset, 'indices'):
        indices = dataset.indices() if callable(dataset.indices) else dataset.indices
    else:
        indices = np.arange(len(dataset))

    candidates = []
    for i in indices:
        batch = full_ds.get(i)
        w = batch.weather[0]
        is_midnight = torch.abs(w[_W_COS_TOD] - 1.0) < 0.01 and torch.abs(w[_W_SIN_TOD]) < 0.01
        is_monday = torch.abs(w[_W_COS_DOW] - 1.0) < 0.01 and torch.abs(w[_W_SIN_DOW]) < 0.01
        if is_midnight and (not args.weekly or is_monday):
            candidates.append(i)

    # Load scaler once; reuse across collect_results calls
    scaler = joblib.load(Path(cfg.earne_data.processed_root) / 'scaler.pkl')

    selected_idx = active_diverse_selection(dataset, args.node, candidates, num_steps, k=2)
    results = [collect_results(model, dataset, args.node, idx, num_steps, device, scaler)
               for idx in selected_idx]
    plot_multi(results, args.out, args.weekly)
