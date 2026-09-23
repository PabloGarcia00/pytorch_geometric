"""
backfill_transform_files.py

One-time backfill: copies each run's fitted normalization file
(transform_{dual,single}.pt) from its processed dataset cache
(datasets/<processed_root>/, gitignored and machine-local) into its results
run dir (results/<run_name>/), alongside the checkpoint(s) it belongs to.

Why this exists: a checkpoint's outputs are normalized-space quantiles --
without the exact transform that normalized its training data, they can't
be converted back into physical Watts. That transform previously lived only
under datasets/, which never travels with results/ (excluded by
.gitignore, and per investigation, not reliably present outside the
original training box either). custom_graphgym/loader/graph_dataset.py now
prefers a results/<run>/transform_*.pt copy over the datasets/ one when
present (see its "Transform (for inverse transforms at inference)" comment)
-- this script backfills that copy for every run trained before that fix
landed. main.py now writes this copy automatically for every new run going
forward (_copy_transform_to_results()), so this script should only need to
run once.

Safety check: a dataset's processed cache can be rebuilt later (e.g. from a
newer raw gold-layer snapshot), which would silently change a same-named
transform file's contents without changing its filename. If that already
happened between a run's earliest checkpoint and today, the transform
currently on disk under datasets/ is NOT the one that checkpoint actually
trained against, and copying it in would create a *confident-looking but
wrong* results-dir file. This script checks each run's transform mtime
against its earliest checkpoint mtime and reports (not silently copies)
any run where the transform is newer -- those need manual judgment, not an
automated copy.

Usage:
    python notebooks/backfill_transform_files.py [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

import yaml

RESULTS_ROOT = Path("results")
DATASETS_ROOT = Path(".")  # processed_root paths in configs are already relative to repo root


def transform_filename(dim_in: int) -> str:
    return "transform_dual.pt" if dim_in == 2 else "transform_single.pt"


def earliest_ckpt_mtime(run_dir: Path) -> float | None:
    ckpts = list(run_dir.glob("*/ckpt/*.ckpt"))
    if not ckpts:
        return None
    return min(c.stat().st_mtime for c in ckpts)


def run(dry_run: bool) -> None:
    copied, skipped_exists, missing_source, suspect, no_ckpt = [], [], [], [], []

    for config_path in sorted(RESULTS_ROOT.glob("*/config.yaml")):
        run_dir = config_path.parent
        try:
            with open(config_path) as f:
                run_cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[SKIP] {run_dir}: unreadable config.yaml ({e})")
            continue

        processed_root = run_cfg.get("earne_data", {}).get("processed_root")
        dim_in = run_cfg.get("model", {}).get("dim_in")
        if not processed_root or dim_in is None:
            print(f"[SKIP] {run_dir}: missing earne_data.processed_root or model.dim_in")
            continue

        fname = transform_filename(dim_in)
        src = Path(processed_root) / fname
        dst = run_dir / fname

        if dst.exists():
            skipped_exists.append(run_dir)
            continue
        if not src.exists():
            missing_source.append((run_dir, src))
            continue

        ckpt_mtime = earliest_ckpt_mtime(run_dir)
        if ckpt_mtime is None:
            no_ckpt.append(run_dir)
            continue

        transform_mtime = src.stat().st_mtime
        is_suspect = transform_mtime > ckpt_mtime
        if is_suspect:
            suspect.append((run_dir, src))
            gap_days = (transform_mtime - ckpt_mtime) / 86400
            print(
                f"[SUSPECT] {run_dir}: {src} was modified {gap_days:.1f} days AFTER "
                f"this run's earliest checkpoint -- the dataset cache was likely "
                f"rebuilt since training. Copying anyway (best available), marked."
            )

        if not dry_run:
            shutil.copy2(src, dst)
            if is_suspect:
                import datetime
                marker = run_dir / "SUSPECT_STALE_TRANSFORM.txt"
                marker.write_text(
                    "This run's transform file was backfilled from "
                    f"{src}, but that file's own mtime "
                    f"({datetime.datetime.fromtimestamp(transform_mtime, tz=datetime.timezone.utc).isoformat()}) "
                    "is AFTER this run's earliest checkpoint's mtime "
                    f"({datetime.datetime.fromtimestamp(ckpt_mtime, tz=datetime.timezone.utc).isoformat()}), "
                    f"a gap of {gap_days:.1f} days.\n\n"
                    "This means the datasets/ processed cache was likely rebuilt "
                    "(e.g. from an updated raw gold-layer snapshot) after this run "
                    "was trained. The transform_*.pt copied into this directory is "
                    "the best available, but it may NOT be the exact normalization "
                    "this checkpoint actually trained under -- denormalized "
                    "(physical-unit) outputs for this run should not be trusted "
                    "without independent verification (e.g. re-deriving expected "
                    "physical-unit ranges from raw data, or retraining).\n\n"
                    "Generated by notebooks/backfill_transform_files.py.\n"
                )
        copied.append(run_dir)

    print()
    print("=== Summary ===")
    print(f"copied{' (dry-run, not written)' if dry_run else ''}: {len(copied)}")
    print(f"  of which suspect (transform newer than earliest checkpoint): {len(suspect)}")
    print(f"already had a results-dir transform (skipped): {len(skipped_exists)}")
    print(f"no matching source transform found under datasets/: {len(missing_source)}")
    print(f"no checkpoint found (skipped mtime check): {len(no_ckpt)}")
    if missing_source:
        print("\nRuns with no source transform (datasets/ cache likely gone):")
        for run_dir, src in missing_source:
            print(f"  {run_dir} -> expected {src}")
    if suspect:
        print("\nSuspect runs needing manual review:")
        for run_dir, src in suspect:
            print(f"  {run_dir} -> {src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report only, don't copy")
    args = parser.parse_args()
    run(args.dry_run)
