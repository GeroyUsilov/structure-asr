#!/usr/bin/env bash
# CPU-only replay from stored results; never loads ESM2 or ProstT5.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON_BIN" -m structure_asr.benchmark_examples
"$PYTHON_BIN" -m structure_asr.score_placement \
  --example-root examples/ef_tu --mapping results/reproduced/node_mapping.tsv \
  --outdir results/score_placement --samples 5000
"$PYTHON_BIN" -m structure_asr.example_distances
"$PYTHON_BIN" examples/glugln/run_corrected_glugln.py --validate-only
"$PYTHON_BIN" examples/glugln/plot_glugln_model_entropies.py
"$PYTHON_BIN" examples/glugln/plot_glugln_sequence_logos.py
