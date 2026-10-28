#!/usr/bin/env bash
# Fetch the CC0 Q.3Di.AF / Q.3Di.LLM matrices (Garg & Hochberg 2025, MBE 42:msaf124;
# data doi:10.17617/3.1MJJBH) into .cache/models/3di and verify published MD5s.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT_DIR/.cache/models/3di"
mkdir -p "$DEST"
fetch() {
  local name="$1" id="$2" md5="$3"
  if [ ! -f "$DEST/$name" ]; then
    curl -sSL -o "$DEST/$name.tmp" "https://edmond.mpg.de/api/access/datafile/$id"
    mv "$DEST/$name.tmp" "$DEST/$name"
  fi
  local actual
  actual="$(python3 -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "$DEST/$name")"
  if [ "$actual" != "$md5" ]; then
    echo "MD5 mismatch for $name: $actual != $md5" >&2
    exit 1
  fi
  echo "$DEST/$name OK"
}
fetch Q.3Di.AF 311466 8f897fffd14f247afbee44dad2fce2ce
fetch Q.3Di.LLM 311467 3a6c99bdf5cc5fc72ba5eaedfc2dd2bd
