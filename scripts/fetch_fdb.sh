#!/usr/bin/env bash
# Fetch Full-Duplex-Bench at the pinned commit into third_party/ (idempotent).
# Benchmark audio is NOT downloaded here: fetch it from the Google Drive link in
# third_party/Full-Duplex-Bench/v3/README.md and extract to v3/fdb_v3_data_released/.
set -euo pipefail

FDB_REPO="https://github.com/DanielLin94144/Full-Duplex-Bench"
FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/Full-Duplex-Bench"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none "$FDB_REPO" "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin "$FDB_COMMIT" 2>/dev/null || git -C "$DEST" fetch origin
git -C "$DEST" -c advice.detachedHead=false checkout "$FDB_COMMIT"
echo "FDB-v3 at $(git -C "$DEST" rev-parse HEAD)"

DATA="$DEST/v3/fdb_v3_data_released"
if [ -d "$DATA" ]; then
  echo "Benchmark data: $(find "$DATA" -name input.wav | wc -l) input.wav files"
else
  echo "Benchmark data missing: download it (see v3/README.md) into $DATA" >&2
fi
