#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

JOBS=(
  smarteole:42 smarteole:43 smarteole:44 smarteole:45 smarteole:46
  sdwpf:42 sdwpf:43 sdwpf:44 sdwpf:45 sdwpf:46
  hai:42 hai:43 hai:44 hai:45 hai:46
)

worker() {
  local gpu="$1" parity="$2" index job dataset seed
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${JOBS[$index]}"
    dataset="${job%%:*}"
    seed="${job##*:}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u \
      scripts/run_generic_state_control_validation.py \
      --dataset "$dataset" --seed "$seed" --device cuda --resume
  done
}

echo "[$(date '+%F %T')] generic-state validation control starts; holdout forbidden"
worker 0 0 2>&1 | tee logs/generic_state_control_validation_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/generic_state_control_validation_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "generic-state controls failed; rerun this resume-safe launcher"
  exit "$STATUS"
fi
python scripts/aggregate_generic_state_control_validation.py \
  2>&1 | tee logs/generic_state_control_validation_aggregate.log
echo "[$(date '+%F %T')] generic-state validation control complete; HOLDOUT WAS NOT EVALUATED"
