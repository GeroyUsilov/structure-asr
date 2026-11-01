#!/usr/bin/env bash
# Run every asr-grid inference stage for one shard on a bootstrapped pod.
#   bash scripts/runpod/run_shard.sh grids/example.toml 0/4 [--max-batch 64] [--model-batch-size 32]
# Stages resume, so rerunning after an interruption continues where it stopped.
# Writes $WORKSPACE/status/{RUNNING,DONE,FAILED} with the stage and exit code.
# `evaluate` (the only stage that reads truth) runs locally after fleet.py sync.
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
WORK="${WORKSPACE:-/workspace}"
GRID="$1"; SHARD="$2"; shift 2
MAX_BATCH=64; MODEL_BATCH=32
while [ "$#" -gt 0 ]; do
  case "$1" in
    --max-batch) MAX_BATCH="$2"; shift 2 ;;
    --model-batch-size) MODEL_BATCH="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
# shellcheck source=/dev/null
source "$WORK/env.sh"
STATUS="$WORK/status"
mkdir -p "$STATUS"
rm -f "$STATUS/DONE" "$STATUS/FAILED"

stage() {
  local name="$1"; shift
  echo "$name $(date -u +%FT%TZ)" > "$STATUS/RUNNING"
  echo "=== $name $(date -u +%FT%TZ)"
  local started=$(date +%s)
  bash scripts/grid.sh "$name" "$GRID" --shard "$SHARD" "$@"
  local code=$?
  echo "=== $name exit $code after $(( $(date +%s) - started )) s"
  if [ "$code" -ne 0 ]; then
    echo "$name exit $code $(date -u +%FT%TZ)" > "$STATUS/FAILED"
    rm -f "$STATUS/RUNNING"
    exit "$code"
  fi
}

stage forward --max-batch "$MAX_BATCH"
stage encode --batch-size "$MODEL_BATCH" --resume
stage asr --jobs "$(nproc)"
if grep -q '^\[sample\]' "$GRID"; then
  stage sample --model-batch-size "$MODEL_BATCH" --resume
fi
bash scripts/grid.sh status "$GRID"
date -u +%FT%TZ > "$STATUS/DONE"
rm -f "$STATUS/RUNNING"
