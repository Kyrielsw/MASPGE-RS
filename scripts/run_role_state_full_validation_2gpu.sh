#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=(42 43 44 45 46)

worker() {
  local gpu="$1" parity="$2" index seed
  for index in "${!SEEDS[@]}"; do
    (( index % 2 == parity )) || continue
    seed="${SEEDS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_state_screen.py \
      --config configs/smarteole_role_state_full_validation_v1.json \
      --output-dir results/role_state_v1_full_validation \
      --checkpoint-dir checkpoints/role_state_v1_full_validation \
      --seed "$seed" --device cuda --resume
  done
}

echo "[$(date '+%F %T')] Experiment 041 full-data role-state validation starts"
worker 0 0 2>&1 | tee logs/role_state_full_validation_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/role_state_full_validation_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "Experiment 041 failed; rerun the resume-safe launcher"
  exit "$STATUS"
fi
python scripts/aggregate_role_state_full_validation.py \
  2>&1 | tee logs/role_state_full_validation_aggregate.log
echo "[$(date '+%F %T')] Experiment 041 complete; HOLDOUT WAS NOT EVALUATED"
