#!/usr/bin/env bash
# Run one asr-grid stage from the repository root, e.g.
#   bash scripts/grid.sh plan grids/example.toml
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python}" -m structure_asr.grid "$@"
