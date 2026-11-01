#!/usr/bin/env bash
# Copy a grid run's small provenance and summary files into experiments/<name>/.
#   bash scripts/record_run.sh grids/my_grid.toml 2026-11-02_my-run [--outdir DIR]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [ "$#" -lt 2 ]; then
  sed -n 2,3p "$0" >&2
  exit 2
fi
GRID="$1"; NAME="$2"; shift 2
OUTDIR=""
if [ "${1:-}" = "--outdir" ]; then OUTDIR="$2"; fi
if [ -z "${PYTHON:-}" ] && [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
SOURCE="$(PYTHONPATH="$PROJECT_DIR/src" "${PYTHON:-python}" -c '
import sys
from pathlib import Path
from structure_asr.grid import load_grid
print(load_grid(Path(sys.argv[1]), Path(sys.argv[2]) if sys.argv[2] else None).outdir)
' "$GRID" "$OUTDIR")"
DEST="experiments/$NAME"
if [ -e "$DEST" ]; then echo "$DEST already exists" >&2; exit 1; fi
[ -d "$SOURCE" ] || { echo "no grid output at $SOURCE" >&2; exit 1; }
mkdir -p "$DEST"
cp "$GRID" "$DEST/"
cd "$SOURCE"
find . \( -path './grid_plan.json' -o -name 'batch_timings_*.tsv' -o -name 'sample_timings_*.tsv' \
  -o -path './evaluation/*_summary.tsv' -o -path './evaluation/node_mapping.tsv' \
  -o -path './*/simulation/manifest.json' -o -path './*/three_di/manifest.json' \
  -o -path './*/asr/*.asr_manifest.json' -o -path './*/sample/*/run_manifest.json' \) \
  -type f -print0 | while IFS= read -r -d '' file; do
    mkdir -p "$PROJECT_DIR/$DEST/$(dirname "$file")"
    cp "$file" "$PROJECT_DIR/$DEST/$file"
  done
cd "$PROJECT_DIR"
if [ ! -e "$DEST/NOTES.md" ]; then
  printf '# %s\n\n- Question:\n- Code: commit %s\n- Hardware:\n- Raw data: %s\n- Deviations:\n\n## Findings\n\n' \
    "$NAME" "$(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')" "$SOURCE" > "$DEST/NOTES.md"
fi
du -sh "$DEST"
echo "Recorded $SOURCE -> $DEST; fill in $DEST/NOTES.md"
