#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
SEEDS=(42 43 44 45 46)
MODES=(dlinear_power fouriergnn_power_official_small itransformer_power softs_power_compact)
JOBS=()
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do JOBS+=("$seed $mode"); done
done

worker() {
  local gpu="$1" parity="$2" index seed mode
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    read -r seed mode <<< "${JOBS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_hai_external_holdout.py \
      --seed "$seed" --mode "$mode" --device cuda --resume
  done
}

echo "[$(date '+%F %T')] HAI frozen external holdout starts; training forbidden"
worker 0 0 2>&1 | tee logs/hai_external_holdout_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/hai_external_holdout_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "HAI external holdout failed; rerun the resume-safe evaluator"
  exit "$STATUS"
fi
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_hai_persistence.py \
    --seed "$seed" --split holdout --device cuda --resume
done
python scripts/aggregate_hai_complete_paper_table.py \
  2>&1 | tee logs/hai_complete_paper_table_aggregate.log
echo "[$(date '+%F %T')] HAI seven-method paper table complete; training was forbidden"
