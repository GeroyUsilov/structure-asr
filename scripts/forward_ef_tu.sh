#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
"${PYTHON:-python}" -m structure_asr.forward \
  --tree examples/ef_tu/reference/AB_eftu_rerooted.tre \
  --alignment examples/ef_tu/reference/EFTu_all_alignment.fasta "$@"
