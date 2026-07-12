#!/usr/bin/env bash

# Run the same grid sweep across multiple base configs (e.g. the GNN plus
# all baselines) in one command, so results are directly comparable.
# Usage: MODELS="earne_phys_single baseline_mlp baseline_lstm baseline_cvae" \
#        GRID=sweep_weather_only bash run_batch_models.sh
# Each $CONFIG is handed to run_batch.sh unmodified -- no changes to
# run_batch.sh/parallel.sh/agg_batch.py.

MODELS=${MODELS:-"earne_phys_single baseline_mlp baseline_lstm baseline_cvae"}
GRID=${GRID:-sweep_weather_only}
export REPEAT=${REPEAT:-3}
export MAX_JOBS=${MAX_JOBS:-8}
export SLEEP=${SLEEP:-1}
export MAIN=${MAIN:-main}

for m in $MODELS; do
  echo "=== $m x $GRID ==="
  CONFIG=$m GRID=$GRID bash run_batch.sh
done
