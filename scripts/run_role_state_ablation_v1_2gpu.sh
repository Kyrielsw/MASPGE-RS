#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=(42 43 44 45 46)
JOBS=()
for mode in shared_units unscaled_history; do
  for seed in "${SEEDS[@]}"; do JOBS+=("smarteole:$mode:$seed"); done
done
for seed in "${SEEDS[@]}"; do JOBS+=("hai:without_a4:$seed"); done

worker() {
  local gpu="$1" parity="$2" index job dataset mode seed
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${JOBS[$index]}"
    IFS=: read -r dataset mode seed <<< "$job"
    if [[ "$dataset" == "smarteole" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_state_screen.py \
        --config configs/smarteole_role_state_full_validation_v1.json \
        --output-dir "results/role_state_ablation_v1/smarteole_${mode}" \
        --checkpoint-dir "checkpoints/role_state_ablation_v1/smarteole_${mode}" \
        --ablation-mode "$mode" --seed "$seed" --device cuda --resume
    else
      CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_hai_role_state_formal.py \
        --output-dir results/role_state_ablation_v1/hai_without_a4 \
        --checkpoint-dir checkpoints/role_state_ablation_v1/hai_without_a4 \
        --ablation-mode without_a4 --seed "$seed" --device cuda --resume
    fi
  done
}

echo "[$(date '+%F %T')] role-state structural ablation validation starts"
echo "15 trainable jobs; HOLDOUT IS FORBIDDEN"
worker 0 0 2>&1 | tee logs/role_state_ablation_v1_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/role_state_ablation_v1_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "Ablation validation failed; rerun this resume-safe launcher"
  exit "$STATUS"
fi
python scripts/aggregate_role_state_ablation_v1.py \
  2>&1 | tee logs/role_state_ablation_v1_aggregate.log
echo "[$(date '+%F %T')] role-state structural ablation validation complete"
echo "HOLDOUT WAS NOT EVALUATED"
