"""
export_profiles.py

Backend half of the two-step JSON -> Typst publication-figure pipeline
(models_profile.typ / physics_profile.typ render the frontend half, reading
whatever this writes). Deliberately picks exactly one seed per model/tier --
the one with the lowest test_loss (this project's checkpoint-selection
metric throughout) -- never a seed-averaged profile, since an average isn't
a trajectory any real checkpoint actually produced.

Two entry points per mode: a single-series export (one --config, or one
--model) and a "-batch" export that first picks ONE shared household+day
across every series being compared, then reuses it for each -- needed for
genuine cross-comparability (e.g. every model's physics figure showing the
same household on the same day, not each independently picking whichever
household best showcases itself).

Usage:
    python visualization/export_profiles.py --mode models \\
        --config dual_wx_off --days 2 --out visualization/data/models.json
    python visualization/export_profiles.py --mode physics --model MLP \\
        --household auto:violation --days 2 --out visualization/data/physics_mlp.json
    python visualization/export_profiles.py --mode models-batch \\
        --days 2 --out visualization/data
    python visualization/export_profiles.py --mode physics-batch \\
        --days 2 --out visualization/data
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Must win over notebooks/ for "import trajectory" -- a legacy, unrelated
# script used to live at notebooks/trajectory.py (now archived to
# results_archives/legacy_notebooks_scripts/) and caused exactly this
# collision; keeping this insert order is still the right defensive habit
# even now that it's gone, in case anything else ever lands a same-named
# module in notebooks/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import custom_graphgym  # noqa — registers loaders/losses/metrics
from calculate_metrics import discover_manifest
from trajectory import (
    CONFIGS,
    MODEL_CONFIGS,
    STEPS_PER_DAY,
    find_violation_window_real,
    find_window_by_date,
    find_window_start,
    iso_local_date,
    load_named_run,
    physmask_run,
    rank_models_by_mae,
    select_household,
    select_household_real_violation,
)

PRIMARY_MODEL = "ST-GNN"


def default_models(manifest, config: str) -> list[str]:
    """
    Default --mode models selection: ST-GNN plus whichever other model has
    the best (lowest) test_mae_pv for this --config -- max 2 models per
    figure, per user request, rather than every model in MODEL_CONFIGS.
    """
    others = rank_models_by_mae(manifest, config, exclude={PRIMARY_MODEL})
    if not others:
        raise SystemExit(f"No non-{PRIMARY_MODEL} model has a runnable checkpoint for --config {config!r}.")
    return [PRIMARY_MODEL, others[0]]


def _series(res: dict, key: str, uid: str, start: int, end: int) -> np.ndarray:
    j = res["user_ids"].index(uid)
    return res[key][j, start:end]


def _iso(timestamps: list) -> list:
    return [t.isoformat() for t in timestamps]


def _load_models_run_results(manifest, config: str, models: list[str]) -> dict:
    read_mode, wx = CONFIGS[config]
    run_results = {}
    for label in models:
        sweep, run_prefix = MODEL_CONFIGS[label][read_mode]
        run_name = f"{run_prefix}-wx={wx}"
        res = load_named_run(manifest, sweep, run_name, "pv", best_seed=True)
        if res is not None:
            run_results[label] = res
    return run_results


def _write_models_json(config: str, uid: str, run_results: dict, start: int, days: int, out_path: str) -> None:
    ref = next(iter(run_results.values()))
    steps = days * STEPS_PER_DAY
    end = min(start + steps, len(ref["timestamps"]))
    timestamps = ref["timestamps"][start:end]

    payload = {
        "config": config,
        "household": uid,
        "date": timestamps[0].date().isoformat(),
        "timestamps": _iso(timestamps),
        "true_pv": _series(ref, "real", uid, start, end).tolist(),
        "models": {
            label: {
                "pv_q50": _series(res, "q50", uid, start, end).tolist(),
                "seed": res["seed"],
                "test_loss": res["test_loss"],
            }
            for label, res in run_results.items()
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    print(f"[OK] wrote {out_path}  (household={uid}, models={sorted(run_results)})")


def export_models(
    config: str, models: list | None, household: str, days: int, out_path: str, window_date: str | None = None,
) -> None:
    manifest = discover_manifest()
    wanted = models or default_models(manifest, config)
    unknown = set(wanted) - set(MODEL_CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown model(s): {sorted(unknown)!r}; choices are {sorted(MODEL_CONFIGS)!r}")

    run_results = _load_models_run_results(manifest, config, wanted)
    if not run_results:
        raise SystemExit(f"No requested model has a runnable checkpoint for --config {config!r}.")

    if window_date is not None:
        uid = household
    elif household.startswith("auto:"):
        uid = select_household(run_results, household.split(":", 1)[1])
    else:
        uid = household

    ref = next(iter(run_results.values()))
    start = (
        find_window_by_date(ref["timestamps"], window_date, days)
        if window_date is not None
        else find_window_start(ref["timestamps"], days)
    )
    _write_models_json(config, uid, run_results, start, days, out_path)


def export_models_batch(configs: list[str], days: int, out_dir: str, household: str = "auto:median") -> None:
    """
    Picks ONE shared household+day (from the first config in `configs`,
    using its default 2-model selection) and reuses it across every config
    -- so the 4 --config figures are genuinely cross-comparable instead of
    each independently landing on a different household/day.
    """
    manifest = discover_manifest()
    ref_config = configs[0]
    wanted = default_models(manifest, ref_config)
    run_results = _load_models_run_results(manifest, ref_config, wanted)
    if not run_results:
        raise SystemExit(f"No requested model has a runnable checkpoint for --config {ref_config!r}.")

    if household.startswith("auto:"):
        uid = select_household(run_results, household.split(":", 1)[1])
    else:
        uid = household
    ref = next(iter(run_results.values()))
    start = find_window_start(ref["timestamps"], days)
    date = iso_local_date(ref["timestamps"][start])
    print(f"[shared] household={uid} date={date} (models={sorted(wanted)}, reference config={ref_config!r})")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for config in configs:
        out_path = str(Path(out_dir) / f"models_{config}.json")
        if config == ref_config:
            _write_models_json(config, uid, run_results, start, days, out_path)
        else:
            export_models(config, wanted, uid, days, out_path, window_date=date)


def _load_physics_tiers(manifest, model: str) -> dict:
    nomask_sweep, nomask_run_prefix = MODEL_CONFIGS[model]["dual"]
    nomask_run = f"{nomask_run_prefix}-wx=False"
    physmask_sweep = MODEL_CONFIGS[model]["physmask"]
    tiers = {
        "no_mask_no_loss": load_named_run(manifest, nomask_sweep, nomask_run, "pv", best_seed=True),
        "mask_only": load_named_run(
            manifest, physmask_sweep, physmask_run(physmask_sweep, "0.0"), "pv", best_seed=True
        ),
        "mask_and_loss": load_named_run(
            manifest, physmask_sweep, physmask_run(physmask_sweep, "0.3"), "pv", best_seed=True
        ),
    }
    missing = [k for k, v in tiers.items() if v is None]
    if missing:
        raise SystemExit(f"--mode physics for {model!r}: missing tier(s) {missing!r} (no runnable checkpoint).")
    return tiers


def _write_physics_json(model: str, uid: str, tiers: dict, start: int, days: int, out_path: str) -> None:
    ref = tiers["no_mask_no_loss"]
    steps = days * STEPS_PER_DAY
    end = min(start + steps, len(ref["timestamps"]))
    timestamps = ref["timestamps"][start:end]
    export_true = np.clip(-_series(ref, "true_net", uid, start, end), 0, None)

    payload = {
        "model": model,
        "household": uid,
        "date": timestamps[0].date().isoformat(),
        "timestamps": _iso(timestamps),
        "true_pv": _series(ref, "real", uid, start, end).tolist(),
        "export_true": export_true.tolist(),
        "tiers": {
            tier: {
                "pv_q50": _series(res, "q50", uid, start, end).tolist(),
                "seed": res["seed"],
                "test_loss": res["test_loss"],
            }
            for tier, res in tiers.items()
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    print(f"[OK] wrote {out_path}  (household={uid}, model={model})")


def export_physics(
    model: str, household: str, days: int, out_path: str, window_date: str | None = None,
) -> None:
    manifest = discover_manifest()
    tiers = _load_physics_tiers(manifest, model)

    if window_date is not None:
        uid = household
    elif household.startswith("auto:violation"):
        parts = household.split(":")
        rank = int(parts[2]) if len(parts) > 2 else 1
        uid = select_household_real_violation(tiers["no_mask_no_loss"], rank)
    elif household.startswith("auto:"):
        uid = select_household(tiers, household.split(":", 1)[1])
    else:
        uid = household

    ref = tiers["no_mask_no_loss"]
    start = (
        find_window_by_date(ref["timestamps"], window_date, days)
        if window_date is not None
        else find_violation_window_real(ref, uid, days)
    )
    _write_physics_json(model, uid, tiers, start, days, out_path)


def export_physics_batch(models: list[str], days: int, out_dir: str, rank: int = 1) -> None:
    """
    Picks ONE shared household+day from a single reference model's ground-
    truth data (real PV vs. the physical export floor is a property of the
    DATA, not of any model's prediction -- identical across every model
    sharing the same test split), then reuses it for every model in
    `models`. Only the reference model's no-mask/no-loss tier needs to be
    loaded before selection happens -- every other model's 3 tiers are
    loaded and written one at a time afterward, so progress is visible
    incrementally (one "[OK] wrote ..." per model) instead of the whole
    batch going silent until every model's inference finishes.
    """
    manifest = discover_manifest()
    ref_model = PRIMARY_MODEL if PRIMARY_MODEL in models else models[0]
    ref_tiers = _load_physics_tiers(manifest, ref_model)
    ref = ref_tiers["no_mask_no_loss"]

    uid = select_household_real_violation(ref, rank)
    start = find_violation_window_real(ref, uid, days)
    date = iso_local_date(ref["timestamps"][start])
    print(f"[shared] household={uid} date={date} (rank={rank}, from {ref_model!r}'s ground truth)")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for model in models:
        tiers = ref_tiers if model == ref_model else _load_physics_tiers(manifest, model)
        out_path = str(Path(out_dir) / f"physics_{model}.json")
        _write_physics_json(model, uid, tiers, start, days, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export best-seed disaggregation profiles to JSON for the Typst renderer."
    )
    parser.add_argument("--mode", choices=["models", "physics", "models-batch", "physics-batch"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default=None, help="--mode physics")
    parser.add_argument(
        "--models", default=None,
        help="--mode models: comma-separated subset (default: ST-GNN + best other model by test_mae_pv); "
             "--mode physics-batch: comma-separated subset (default: all)",
    )
    parser.add_argument("--config", choices=sorted(CONFIGS), default="dual_wx_off", help="--mode models")
    parser.add_argument(
        "--configs", default=None, help="--mode models-batch: comma-separated subset (default: all)"
    )
    parser.add_argument(
        "--household",
        default="auto:median",
        help="'auto:best|median|worst' or explicit user_id (models); 'auto:violation[:<rank>]' (physics)",
    )
    parser.add_argument("--rank", type=int, default=1, help="--mode physics-batch: pooled violation rank")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--out", required=True, help="output file (models/physics) or directory (*-batch)")
    args = parser.parse_args()

    if args.mode == "models":
        export_models(
            args.config,
            args.models.split(",") if args.models else None,
            args.household,
            args.days,
            args.out,
        )
    elif args.mode == "models-batch":
        configs = args.configs.split(",") if args.configs else sorted(CONFIGS)
        export_models_batch(configs, args.days, args.out)
    elif args.mode == "physics":
        if args.model is None:
            raise SystemExit("--mode physics needs --model <name>")
        export_physics(args.model, args.household, args.days, args.out)
    else:  # physics-batch
        models = args.models.split(",") if args.models else sorted(MODEL_CONFIGS)
        export_physics_batch(models, args.days, args.out, rank=args.rank)
