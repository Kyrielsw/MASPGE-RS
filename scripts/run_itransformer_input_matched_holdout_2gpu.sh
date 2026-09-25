#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
worker() {
  local gpu="$1"; shift
  for job in "$@"; do
    IFS=: read -r dataset seed <<< "$job"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_itransformer_input_matched_holdout.py \
      --dataset "$dataset" --seed "$seed" --device cuda --resume \
      2>&1 | tee "logs/itransformer_input_matched_holdout_${dataset}_seed${seed}.log"
  done
}
jobs0=(smarteole:42 smarteole:44 smarteole:46 sdwpf:43 sdwpf:45 hai:42 hai:44 hai:46 xai4heat:43 xai4heat:45)
jobs1=(smarteole:43 smarteole:45 sdwpf:42 sdwpf:44 sdwpf:46 hai:43 hai:45 xai4heat:42 xai4heat:44 xai4heat:46)
worker 0 "${jobs0[@]}" & p0=$!
worker 1 "${jobs1[@]}" & p1=$!
wait "$p0"; wait "$p1"
python scripts/aggregate_itransformer_input_matched.py --include-holdout 2>&1 | tee logs/itransformer_input_matched_holdout_aggregate.log
echo "FROZEN POST-HOC HOLDOUT REPORT COMPLETE; NO TRAINING PERFORMED"
