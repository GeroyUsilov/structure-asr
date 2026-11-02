#!/usr/bin/env bash
# Prepare a RunPod pod for asr-grid (idempotent; rerun after an interruption).
# Run inside the checkout that fleet.py cloned from its git bundle, on an image
# that provides CUDA PyTorch (e.g. runpod/pytorch). Everything lands on /workspace:
#   /workspace/venv   venv over the image's torch (--system-site-packages)
#   /workspace/hf     HF_HOME with ESM2 and ProstT5 weights
#   /workspace/env.sh exports for later stages (sourced by run_shard.sh)
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
WORK="${WORKSPACE:-/workspace}"
ESM2_MODEL="${ESM2_MODEL:-facebook/esm2_t33_650M_UR50D}"
PROSTT5_MODEL="${PROSTT5_MODEL:-Rostlab/ProstT5}"
IQTREE_VERSION=3.1.4
export HF_HOME="$WORK/hf" HF_HUB_ENABLE_HF_TRANSFER=1
started=$(date +%s)

if [ ! -x "$WORK/venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$WORK/venv"
fi
PY="$WORK/venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements-gpu.txt
"$PY" -m pip install -q --no-deps -e .
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA is not available in this pod"
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}", flush=True)
EOF

# Weights: ESM2 through the same from_pretrained call the forward stage uses;
# ProstT5 through structure_asr.prostt5.materialize_model (the sampler's path).
"$PY" - "$ESM2_MODEL" "$PROSTT5_MODEL" <<'EOF'
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
from structure_asr.prostt5 import materialize_model
AutoTokenizer.from_pretrained(sys.argv[1])
AutoModelForMaskedLM.from_pretrained(sys.argv[1])
print("ProstT5 snapshot:", materialize_model(sys.argv[2]), flush=True)
EOF

bash scripts/fetch_3di_matrix.sh

case "$(uname -m)" in
  x86_64) flavor=Linux-intel; sha=d422cb2b8f04825faea753afda25c60de6611537d07ec8fcd0033e70cb042839 ;;
  aarch64) flavor=Linux-arm; sha=48fbbee1a5b5eb5b905d3bb9b88ed7651feaa83e9239f66559e3cc3a03455e37 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
TOOLS="$PROJECT_DIR/.cache/tools"
IQTREE="$TOOLS/iqtree-$IQTREE_VERSION-$flavor/bin/iqtree3"
if [ ! -x "$IQTREE" ]; then
  mkdir -p "$TOOLS"
  tarball="$TOOLS/iqtree-$IQTREE_VERSION-$flavor.tar.gz"
  curl -sSL -o "$tarball" \
    "https://github.com/iqtree/iqtree3/releases/download/v$IQTREE_VERSION/iqtree-$IQTREE_VERSION-$flavor.tar.gz"
  echo "$sha  $tarball" | sha256sum -c -
  # Pod /workspace (network FS) refuses chown, so keep the extracting user as owner.
  tar --no-same-owner -xzf "$tarball" -C "$TOOLS"
fi
"$IQTREE" --version | head -1

cat > "$WORK/env.sh" <<EOF
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export PYTHON="$PY"
export ASR_IQTREE="$IQTREE"
EOF
echo "Bootstrap complete in $(( $(date +%s) - started )) s" | tee "$WORK/BOOTSTRAPPED"
