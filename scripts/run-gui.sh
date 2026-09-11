#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DISPLAY="${DISPLAY:-:16}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-box}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
if ! XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" pw-cli info 0 >/dev/null 2>&1; then
  pipewire >/tmp/pw.log 2>&1 &
  sleep 0.4
  wireplumber >/tmp/wp.log 2>&1 &
  sleep 0.4
  pipewire-pulse >/tmp/pp.log 2>&1 &
  sleep 0.6
fi
cd "$ROOT"
exec env DISPLAY="$DISPLAY" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" "$ROOT/target/release/edge-ort-gui" "$@"
