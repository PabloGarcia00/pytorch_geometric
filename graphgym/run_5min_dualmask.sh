#!/usr/bin/env bash

# Trains + scores the previously-missing (dual-read, mask_physics_impossible=True,
# weather_mode=False, require_full_span=True, 5-minute-resolution) cell, for all
# 7 disaggregation architectures. See configs/pyg/res_eval_{arch}_5min_dualmask.yaml.
#
# Dataset processing is automatic on first run (PyG's lazy InMemoryDataset):
# whichever architecture launches first for a given dataset dir builds it from
# the EFS gold-layer file (fleet_gold_layer_5min.parquet, 3GB, read directly
# over NFS -- no copy-to-instance-store step, but requires the EFS mount
#   /mnt/custom-file-systems/efs/fs-0e26e28a2c1df7f40/eda-gold-layer/
# to be attached on this instance). Confirmed ~1min per dataset build on an
# 8-core CPU instance. Two dataset dirs total, not seven:
#   datasets/earne_cvae_dual_physmask_5min/   (CVAE only -- its own pv
#                                               normalization is minmax, not ihs)
#   datasets/earne_dual_physmask_5min/        (shared: KNN/Linear/SVR/LSTM/MLP/GNN)
#
# KNN is deterministic (non-gradient lookup, see custom_graphgym/train/
# knn_train.py) -- 1 seed only regardless of the plan below.
#
# REVISED PLAN (single-seed timing pass first): running all 3 seeds for
# every gradient-trained architecture up front turned out to be premature --
# CVAE alone took ~57h projected for 3 seeds, with 6 more architectures
# unmeasured. So: run every architecture ONCE first (repeat=1) to get a real
# per-architecture epoch-count + wall-clock-per-epoch data point for all 7,
# then decide whether seeds 2-3 are worth the added time per architecture.
#
# CVAE's single seed is DONE already -- results/res_eval_cvae_5min_dualmask/0/
# (37 epochs, ~31 min/epoch measured wall-clock, ~19h total) -- not rerun
# below. To extend any architecture to more seeds later, don't just rerun
# with --repeat 3: cfg.seed resets to the config's base value each process
# invocation (main.py:48, `cfg.seed = cfg.seed + 1` inside the --repeat
# loop), so a fresh `--repeat 3` call would redo seed 0 from scratch before
# reaching 1/2. Bump the config's base seed (or just accept redoing seed 0,
# cheap relative to the new seeds) if/when that's wanted.
#
# Usage:
#   bash run_5min_dualmask.sh

set -euo pipefail

python main.py --cfg configs/pyg/res_eval_lstm_5min_dualmask.yaml   --repeat 1
python main.py --cfg configs/pyg/res_eval_mlp_5min_dualmask.yaml    --repeat 1
python main.py --cfg configs/pyg/res_eval_gnn_5min_dualmask.yaml    --repeat 1
python main.py --cfg configs/pyg/res_eval_linear_5min_dualmask.yaml --repeat 1
python main.py --cfg configs/pyg/res_eval_svr_5min_dualmask.yaml    --repeat 1
python main.py --cfg configs/pyg/res_eval_knn_5min_dualmask.yaml    --repeat 1

cat <<'EOF'

Training done. Now score in two passes, not one -- CVAE needs GPU to finish
in reasonable time (see note below), the other six don't need it and are
faster run CPU-parallel:

  # 1) CVAE alone, serial mode (--workers 1 is the *only* path in
  #    calculate_metrics.py that doesn't force CPU -- disaggregate_test_set()'s
  #    own cfg.accelerator="cuda" if torch.cuda.is_available() picks up a real
  #    GPU here; --workers >1 always forces CPU via _init_worker(), to stop
  #    many concurrent workers contending for one GPU).
  python notebooks/calculate_metrics.py \
      --run-name-glob 'res_eval_cvae_5min_dualmask' --workers 1

  # 2) Everything else in this batch, CPU-parallel as usual.
  python notebooks/calculate_metrics.py \
      --run-name-glob 'res_eval_*_5min_dualmask' --workers 64

  # (Or fold both into your regular full calculate_metrics.py --workers 64
  #  run -- but then CVAE's row runs CPU-bound like every other worker, and
  #  needs --row-timeout raised to ~50000 (~14h) to survive on CPU. On CPU,
  #  measured ~15s/batch x 2957 batches = ~12.3h just for that one row, vs.
  #  the default 3h --row-timeout. Root cause: baseline_cvae_network.py's own
  #  bidirectional LSTM encoder (line ~126), profiled at 99% of per-batch
  #  time, doesn't have an efficient CPU kernel for a seq_len=288 recurrence
  #  the way cuDNN's fused GPU LSTM kernel does -- this is specific to CVAE,
  #  not the other 6 architectures, which already score fine on CPU alone.)

No calculate_metrics.py/held_out_metrics.py changes are needed for these new
runs to be *discovered* -- their run_name (res_eval_*_5min_dualmask) already
matches the existing "res_eval_*" -> "resolution_eval" sweep glob in
notebooks/calculate_metrics.py's _FLAT_SWEEP_GLOBS, and every stratification
column (dual_read, mask_physics_impossible, weather_mode, require_full_span,
resolution) is read straight from each run's own config.yaml. Only
--run-name-glob (new, opt-in, default off) was added, so this batch can be
scored separately from the other ~470 existing runs without slowing them
down or being slowed down by them.

held_out_metrics.py was intentionally left untouched -- this batch is
in-distribution (require_full_span=True) only, no held-out request was made.
EOF
