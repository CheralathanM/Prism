#!/usr/bin/env bash
# Download the default local Piper voice (offline TTS; no API key, no cost).
# Source: https://huggingface.co/rhasspy/piper-voices (en_US / lessac / medium).
# Override the location with FDAGENT_PIPER_VOICE=/path/to/voice.onnx (the .onnx.json must sit next to it).
set -euo pipefail
DEST="${FDAGENT_PIPER_DIR:-$HOME/.cache/fdagent/piper}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
mkdir -p "$DEST/en/en_US/lessac/medium"
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
  out="$DEST/en/en_US/lessac/medium/$f"
  [ -s "$out" ] && { echo "present: $out"; continue; }
  curl -fL --retry 3 -o "$out.part" "$BASE/$f" && mv "$out.part" "$out"
  echo "downloaded: $out"
done
