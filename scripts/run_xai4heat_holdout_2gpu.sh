#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
SEEDS=(42 43 44 45 46)
MODES=(persistence dlinear_power fouriergnn_power_official_small itransformer_power softs_power_compact timemixer_power mafs_adapted_fully generic_unit_state)
JOBS=()
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do JOBS+=("$seed $mode"); done
done
worker() {
  local gpu="$1" parity="$2" index seed mode
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    read -r seed mode <<< "${JOBS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_xai4heat_holdout.py \
      --seed "$seed" --mode "$mode" --device cuda --resume
  done
}
echo "[$(date '+%F %T')] XAI4HEAT frozen holdout starts; TRAINING FORBIDDEN"
worker 0 0 2>&1 | tee logs/xai4heat_holdout_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/xai4heat_holdout_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "XAI4HEAT holdout failed; rerun this resume-safe launcher"
  exit "$STATUS"
fi
python scripts/aggregate_xai4heat_holdout.py \
  2>&1 | tee logs/xai4heat_holdout_aggregate.log
echo "[$(date '+%F %T')] XAI4HEAT frozen holdout complete"
