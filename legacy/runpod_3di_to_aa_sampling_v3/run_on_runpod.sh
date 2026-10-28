#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

N_SAMPLES="${N_SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
MAX_CHUNK_STATES="${MAX_CHUNK_STATES:-500}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.90}"
TOP_K="${TOP_K:-6}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MODEL="${MODEL:-Rostlab/ProstT5}"
OUTDIR="${OUTDIR:-${HERE}/results_rejection_resume}"
RESUME_FROM="${RESUME_FROM:-/workspace/runpod_3di_to_aa_sampling/results}"

python -m pip install -q -r "${HERE}/requirements-runpod.txt"

python "${HERE}/run_3di_to_aa_sampling.py" \
  --input-dir "${HERE}/inputs" \
  --outdir "${OUTDIR}" \
  --resume-from "${RESUME_FROM}" \
  --n-samples "${N_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --max-chunk-states "${MAX_CHUNK_STATES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --repetition-penalty "${REPETITION_PENALTY}" \
  --model "${MODEL}" \
  --device cuda \
  --dtype float16 \
  --resume
