# Models

Place weights here. Prefer `./scripts/download-models.sh`.

| File | Plug id | Capability | Status |
|------|---------|------------|--------|
| `ggml-tiny.bin` | `whisper-tiny` | asr | **Default** — whisper.cpp multilingual tiny (~75 MB) |
| `ggml-base.bin` | (GUI picker) | asr | whisper.cpp base (~142 MB); `WHISPER_SIZES` default includes base |
| `ggml-small.bin` | (GUI picker) | asr | whisper.cpp small (~466 MB); set `WHISPER_SIZES=… small` |
| `ggml-medium.bin` | (GUI picker) | asr | whisper.cpp medium (~1.5 GB); explicit only |
| `ggml-large-v3.bin` | (GUI picker) | asr | whisper.cpp large-v3 (~3 GB); explicit only |
| `tiny_asr_energy.onnx` | `tiny-asr-energy` | asr | Bundled ORT demo (energy → 4 logits) |
| `silero_vad.onnx` | `silero-vad` | vad | Download script (energy fallback until present) |
| `jfk.wav` | (fixture) | — | Sample speech for `--source file` / tests |
| `en_US-lessac-low.onnx` (+ `.onnx.json`) | `piper-en` | tts | Optional Piper low voice |
| `nllb.onnx` | `nllb-200-distilled` | mt | Large / skipped by download script |
| `whisper-tiny-onnx/` | experimental | asr | Optional Xenova ONNX (`FETCH_WHISPER_ONNX=1`) |

Large `*.onnx` / `*.bin` / `*.wav` under `models/` are gitignored except the bundled tiny ASR demo. Test fixture: `tests/fixtures/jfk.wav`.

## Whisper (default)

Default profile ASR is **`whisper_ggml`** (`profiles/default/asr.whisper.json` → `ggml-tiny.bin`).
The GUI can switch sizes (tiny / base / small / medium / large-v3) via `AsrNode::set_model_path`.

```bash
# Default downloads tiny + base (~75 + ~142 MB)
./scripts/download-models.sh

# Also fetch small (~466 MB)
WHISPER_SIZES="tiny base small" ./scripts/download-models.sh

# then:
cargo run --release --bin listen -- --source file --file models/jfk.wav --no-vad --no-mt
```

Language: GUI picker or `meta.language` — `auto` | `en` | `fr` | `es` | `ar` | `ru` | `hi` | `zh` | `ko` | `ja` | `ur` | `fa` | `tr` | `az` | `hy`.
Task: `meta.task` = `transcribe` (default) or `translate` (speech → English). GUI **Translate → EN** dual-decodes for Original | English panels.

## ORT Whisper (optional)

Set `FETCH_WHISPER_ONNX=1` when running the download script to pull Xenova quantized encoder/decoder.
Point a manifest at those files with `meta.backend = "whisper"`. Full ORT decoder loop is still experimental; prefer ggml for real text.

## Silero VAD

```
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

## Piper

```
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx
```

## NLLB

Skipped — too large. Place manually and update `mt.nllb.json` if needed.
