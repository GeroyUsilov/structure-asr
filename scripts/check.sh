#!/usr/bin/env bash
# Local pre-commit check: CPU test suite, then the CPU example replay.
#   bash scripts/check.sh            # tests + replay
#   bash scripts/check.sh --quick    # tests only
# Uses .venv/bin/python when present (see README), else $PYTHON or python.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [ -z "${PYTHON:-}" ] && [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m unittest discover -s tests
if [ "${1:-}" != "--quick" ]; then
  bash scripts/reproduce_examples.sh >/dev/null
  echo "Example replay passed"
fi
git diff --quiet -- examples || { echo "examples/ was modified" >&2; exit 1; }
