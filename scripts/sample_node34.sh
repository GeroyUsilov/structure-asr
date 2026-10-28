#!/usr/bin/env bash
# From this checkout, run one independent blinded-to-truth sampling chain.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
WEIGHT="${1:-0.02}"
if [ "$#" -gt 0 ]; then shift; fi
"${PYTHON:-python}" -m structure_asr.sampling \
  --aa-state examples/ef_tu/mu1_tau2/aa.state \
  --three-di-state examples/ef_tu/mu1_tau2/3di.state \
  --nodes-file examples/ef_tu/node34.txt \
  --condition mu1_tau2 --outdir "results/node34_lambda_${WEIGHT}" \
  --structural-lambda "$WEIGHT" --n-iter 1000 --chains-per-node 1 \
  --mcmc-temperature 2 --site-uncertainty-mixture 0.9 \
  --model "${PROSTT5_MODEL:-Rostlab/ProstT5}" --resume "$@"
