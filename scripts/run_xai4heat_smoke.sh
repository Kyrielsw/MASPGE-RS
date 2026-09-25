#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_xai4heat_main_validation.py \
  --seed 42 --device cuda --smoke \
  --output-dir results/xai4heat_smoke_v1/main \
  --checkpoint-dir checkpoints/xai4heat_smoke_v1/main \
  2>&1 | tee logs/xai4heat_smoke_main.log
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_external_baseline_formal.py \
  --config configs/xai4heat_external_baselines_v1.json \
  --mode dlinear_power --seed 42 --device cuda --validation-only --smoke \
  --output-dir results/xai4heat_smoke_v1/external \
  --checkpoint-dir checkpoints/xai4heat_smoke_v1/external \
  2>&1 | tee logs/xai4heat_smoke_dlinear.log
echo "[$(date '+%F %T')] XAI4HEAT minimal smoke complete; HOLDOUT WAS NOT EVALUATED"
