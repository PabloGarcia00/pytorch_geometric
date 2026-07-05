#!/usr/bin/env bash
# Runs BiLSTM and CVAE training in parallel, then evaluates both and prints metrics.
set -e

PYTHON="../graphgym/.venv/bin/python3"
OUT="results"

echo "[$(date '+%H:%M:%S')] Starting BiLSTM (50 epochs) ..."
$PYTHON train.py --config configs/bilstm.yaml --output_dir $OUT \
  > $OUT/bilstm_train.log 2>&1 &
PID_BILSTM=$!

echo "[$(date '+%H:%M:%S')] Starting CVAE (50 epochs) ..."
$PYTHON train.py --config configs/cvae.yaml --output_dir $OUT \
  > $OUT/cvae_train.log 2>&1 &
PID_CVAE=$!

echo "BiLSTM PID=$PID_BILSTM  CVAE PID=$PID_CVAE"
echo "Logs: results/bilstm_train.log  results/cvae_train.log"
echo ""

wait $PID_BILSTM
RC_BILSTM=$?
echo "[$(date '+%H:%M:%S')] BiLSTM training done (exit=$RC_BILSTM)"

wait $PID_CVAE
RC_CVAE=$?
echo "[$(date '+%H:%M:%S')] CVAE training done (exit=$RC_CVAE)"

echo ""
echo "[$(date '+%H:%M:%S')] Running evaluations ..."

$PYTHON evaluate.py --config configs/bilstm.yaml --output_dir $OUT \
  > $OUT/bilstm_eval.log 2>&1
echo "[$(date '+%H:%M:%S')] BiLSTM evaluation done"

$PYTHON evaluate.py --config configs/cvae.yaml --output_dir $OUT \
  > $OUT/cvae_eval.log 2>&1
echo "[$(date '+%H:%M:%S')] CVAE evaluation done"

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo ""
echo "--- BiLSTM ---"
python3 -c "import json; m=json.load(open('$OUT/bilstm_metrics.json')); [print(f'  {k}: {v}') for k,v in m.items()]"
echo ""
echo "--- CVAE ---"
python3 -c "import json; m=json.load(open('$OUT/cvae_metrics.json')); [print(f'  {k}: {v}') for k,v in m.items()]"
