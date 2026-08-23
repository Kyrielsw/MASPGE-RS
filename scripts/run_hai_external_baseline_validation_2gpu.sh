#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_external_baseline_formal.py \
      --config configs/hai_external_baselines_v1.json \
      --seed "$seed" --mode "$mode" --device cuda --validation-only --resume \
      --output-dir results/hai_external_validation_v1 \
      --checkpoint-dir checkpoints/hai_external_validation_v1
  done
}

echo "[$(date '+%F %T')] HAI external validation starts; 20 trainable jobs"
worker 0 0 2>&1 | tee logs/hai_external_validation_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/hai_external_validation_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "HAI external validation failed; rerun the resume-safe launcher"
  exit "$STATUS"
fi
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_hai_persistence.py \
    --seed "$seed" --split validation --device cuda --resume
done
python scripts/aggregate_hai_paper_validation.py \
  2>&1 | tee logs/hai_paper_validation_aggregate.log
echo "[$(date '+%F %T')] HAI paper validation complete; HOLDOUT WAS NOT EVALUATED"
