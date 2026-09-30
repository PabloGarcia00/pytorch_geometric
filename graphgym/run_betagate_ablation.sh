#!/usr/bin/env bash

# Beta+gate vs. quantile head ablation, 3 encoders x 2 heads, all on the
# same 15-min/native-cadence, dual-read, weather-off, physics-masked,
# full-span ("optimal") dataset (datasets/earne_dual_physmask). See
# scratch_gen_betagate_ablation_configs.py for how the 6 configs below were
# generated from their earne_learned_corr_dual_physmask.yaml/
# baseline_lstm_dual_physmask.yaml/baseline_mlp_dual_physmask.yaml
# templates -- only head_name/loss_fun differ within each encoder pair.
#
# Single seed each (repeat=1) for a first timing/signal pass, same
# rationale as run_5min_dualmask.sh. Single GPU -> sequential, not
# parallel.
#
# Usage:
#   bash run_betagate_ablation.sh

set -euo pipefail

python main.py --cfg configs/pyg/betagate_gnn_quantile.yaml  --repeat 1
python main.py --cfg configs/pyg/betagate_gnn_betagate.yaml  --repeat 1
python main.py --cfg configs/pyg/betagate_lstm_quantile.yaml --repeat 1
python main.py --cfg configs/pyg/betagate_lstm_betagate.yaml --repeat 1
python main.py --cfg configs/pyg/betagate_mlp_quantile.yaml  --repeat 1
python main.py --cfg configs/pyg/betagate_mlp_betagate.yaml  --repeat 1

cat <<'EOF'

Training done. Score with:
  python notebooks/calculate_metrics.py --run-name-glob 'betagate_*' --workers 64
EOF
