#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
SEEDS=(42 43 44 45 46)

worker() {
  local gpu="$1" parity="$2" index seed
  for index in "${!SEEDS[@]}"; do
    (( index % 2 == parity )) || continue
    seed="${SEEDS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_hai_role_state_holdout.py \
      --seed "$seed" --device cuda --resume
  done
}

echo "[$(date '+%F %T')] strict frozen HAI holdout starts; training is forbidden"
worker 0 0 2>&1 | tee logs/hai_role_state_holdout_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/hai_role_state_holdout_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "HAI holdout report failed; rerun only the resume-safe evaluator"
  exit "$STATUS"
fi
python scripts/aggregate_hai_role_state_holdout.py \
  2>&1 | tee logs/hai_role_state_holdout_aggregate.log
echo "[$(date '+%F %T')] strict HAI holdout report complete; NO POST-HOLDOUT TUNING"
