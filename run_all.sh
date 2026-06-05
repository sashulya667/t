#!/usr/bin/env bash
# Train + infer all three adapters (0.5B, 1.5B, 7B). Run from lora_server_bundle/.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LORA_EVAL_MAX_NEW_TOKENS="${LORA_EVAL_MAX_NEW_TOKENS:-1024}"
python run_pipeline_gpu.py --all-models "$@"
