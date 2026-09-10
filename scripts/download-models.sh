#!/usr/bin/env bash
# Fetch optional model plugs into models/. Cap size/time; failures are non-fatal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"
cd "$MODELS"

TIMEOUT_SECS="${DOWNLOAD_TIMEOUT:-300}"
# ggml-small ~466MB; default cap high enough for small when requested
MAX_MB="${DOWNLOAD_MAX_MB:-600}"
# Default: tiny + base (base ~142MB). Add small/medium/large-v3 explicitly if needed.
WHISPER_SIZES="${WHISPER_SIZES:-tiny base}"

fetch() {
  local url="$1"
  local out="$2"
  local label="$3"
  local max_mb="${4:-$MAX_MB}"
  if [[ -f "$out" ]]; then
    echo "[skip] $label already present: $out ($(du -h "$out" | cut -f1))"
    return 0
  fi
  echo "[fetch] $label (cap ${max_mb}MB)"
  echo "        $url"
  if ! curl -fL --connect-timeout 20 --max-time "$TIMEOUT_SECS" \
      -A "edge-ort-runtime-download/0.1" \
      -o "${out}.partial" "$url"; then
    echo "[fail] $label — download error (see URL above). Continuing."
    rm -f "${out}.partial"
    return 1
  fi
  local bytes
  bytes=$(wc -c < "${out}.partial" | tr -d ' ')
  local max_bytes=$((max_mb * 1024 * 1024))
  if (( bytes > max_bytes )); then
    echo "[fail] $label — file ${bytes}B exceeds ${max_mb}MB cap; removing."
    rm -f "${out}.partial"
    return 1
  fi
  if (( bytes < 1000 )); then
    echo "[fail] $label — file too small (${bytes}B); likely an error page."
    rm -f "${out}.partial"
    return 1
  fi
  mv "${out}.partial" "$out"
  echo "[ok]   $label → $out ($(du -h "$out" | cut -f1))"
}

whisper_size_info() {
  # prints: filename|approx_label|per_file_cap_mb
  case "$1" in
    tiny)     echo "ggml-tiny.bin|Whisper ggml-tiny (~75MB)|120" ;;
    base)     echo "ggml-base.bin|Whisper ggml-base (~142MB)|200" ;;
    small)    echo "ggml-small.bin|Whisper ggml-small (~466MB)|600" ;;
    medium)   echo "ggml-medium.bin|Whisper ggml-medium (~1.5GB)|2000" ;;
    large-v3) echo "ggml-large-v3.bin|Whisper ggml-large-v3 (~3GB)|4000" ;;
    *)        echo "" ;;
  esac
}

echo "=== edge-ort-runtime model download ==="
echo "dest: $MODELS"
echo "timeout=${TIMEOUT_SECS}s  default_max=${MAX_MB}MB"
echo "WHISPER_SIZES=${WHISPER_SIZES}"
echo

# 1) Silero VAD (small, required for real VAD)
SILERO_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
fetch "$SILERO_URL" "silero_vad.onnx" "Silero VAD" 10 || {
  echo "  alt: https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
}

# 2) Whisper ASR — ggml sizes from WHISPER_SIZES (default: tiny base)
echo
GGML_BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
for size in $WHISPER_SIZES; do
  info="$(whisper_size_info "$size")"
  if [[ -z "$info" ]]; then
    echo "[warn] unknown Whisper size '$size' — skipping (known: tiny base small medium large-v3)"
    continue
  fi
  IFS='|' read -r fname label cap <<<"$info"
  # Prefer DOWNLOAD_MAX_MB if set higher than per-file default; else use per-file.
  local_cap="$cap"
  if (( MAX_MB > cap )); then
    local_cap="$MAX_MB"
  fi
  fetch "${GGML_BASE_URL}/${fname}" "$fname" "$label" "$local_cap" || true
done

# Optional: Xenova Whisper-tiny quantized ONNX (for backend=whisper ORT experiments)
if [[ "${FETCH_WHISPER_ONNX:-0}" == "1" ]]; then
  echo
  HF="https://huggingface.co/Xenova/whisper-tiny/resolve/main"
  mkdir -p whisper-tiny-onnx
  fetch "$HF/onnx/encoder_model_quantized.onnx" "whisper-tiny-onnx/encoder_model_quantized.onnx" "Whisper ONNX encoder (quantized)" || true
  fetch "$HF/onnx/decoder_model_merged_quantized.onnx" "whisper-tiny-onnx/decoder_model_merged_quantized.onnx" "Whisper ONNX decoder merged (quantized)" || true
  fetch "$HF/tokenizer.json" "whisper-tiny-onnx/tokenizer.json" "Whisper tokenizer.json" || true
  fetch "$HF/tokenizer_config.json" "whisper-tiny-onnx/tokenizer_config.json" "Whisper tokenizer_config.json" || true
  fetch "$HF/config.json" "whisper-tiny-onnx/config.json" "Whisper config.json" || true
  fetch "$HF/vocab.json" "whisper-tiny-onnx/vocab.json" "Whisper vocab.json" || true
fi

# 3) Sample speech WAV for smoke tests (JFK ~11s)
echo
JFK_URL="https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
fetch "$JFK_URL" "jfk.wav" "Sample jfk.wav (speech fixture)" 5 || true

# 4) Piper voice (optional, low quality = smaller)
echo
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low"
fetch "${PIPER_BASE}/en_US-lessac-low.onnx" "en_US-lessac-low.onnx" "Piper en_US-lessac-low.onnx" 80 || true
fetch "${PIPER_BASE}/en_US-lessac-low.onnx.json" "en_US-lessac-low.onnx.json" "Piper en_US-lessac-low.onnx.json" 5 || true
if [[ -f en_US-lessac-low.onnx && ! -e piper_en.onnx ]]; then
  ln -sf en_US-lessac-low.onnx piper_en.onnx
  echo "[link] piper_en.onnx → en_US-lessac-low.onnx"
fi

# 5) NLLB (large — skip)
echo
echo "[skip] NLLB — large multilingual MT; export separately if needed."
echo "       Place as models/nllb.onnx and keep mt.nllb.json. Passthrough until present."

echo
echo "=== summary ==="
ls -lh "$MODELS" | sed 's/^/  /'
echo
echo "Whisper ggml sizes:"
for size in tiny base small medium large-v3; do
  info="$(whisper_size_info "$size")"
  IFS='|' read -r fname label _cap <<<"$info"
  if [[ -f "$fname" ]]; then
    echo "  ✓ $fname  ($(du -h "$fname" | cut -f1))  — $label"
  else
    echo "  ✗ $fname  — not present ($label)"
  fi
done
echo
echo "Default ASR: whisper_ggml (models/ggml-tiny.bin) via profiles/default/asr.whisper.json"
echo "Bundled demo ORT ASR still available: tiny_asr_energy.onnx → asr.tiny_energy.json"
echo "Fetch more: WHISPER_SIZES=\"tiny base small\" ./scripts/download-models.sh"
