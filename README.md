# edge-ort-runtime

Native **Linux** app for **pluggable PipeWire audio** + **ONNX Runtime** inference, with a CLI (`listen`) and a GUI (`edge-ort-gui`).

Core types are capability-oriented (`vad | asr | mt | tts`). Concrete models — Silero, Whisper, NLLB, Piper, or anything else — are **plugs** described by JSON manifests. They are never baked into core enums beyond the capability kind.

> Sibling project: [`edge-audio-ai`](../edge-audio-ai) (separate codebase; this crate does not depend on it).

## Quick start

```bash
# 1) System deps (Debian/Ubuntu-style)
sudo apt install build-essential pkg-config cmake clang libclang-dev \
  libpipewire-0.3-dev libxkbcommon0 libxkbcommon-x11-0 libxcb-xkb1 \
  pipewire wireplumber pulseaudio-utils

# If bindgen cannot find libclang:
export LIBCLANG_PATH=/usr/lib/llvm-19/lib   # adjust to your LLVM

# 2) Rust 1.85+
rustc --version

# 3) Build
cd edge-ort-runtime
cargo build --release

# 4) Models (default: ggml-tiny + ggml-base + Silero VAD + sample WAV)
./scripts/download-models.sh

# 5) Optional CJK fonts for the GUI (SC/TC/JP/KR) — ~16 MB each
./scripts/extract-cjk-fonts.sh   # needs Noto Sans CJK .ttc on the system

# 6) Run GUI (starts/uses PipeWire; set DISPLAY / XDG_RUNTIME_DIR as needed)
./scripts/run-gui.sh
# or: ./target/release/edge-ort-gui
```

Binaries after `cargo build --release`:

| Binary | Role |
|--------|------|
| `target/release/edge-ort-gui` | Desktop UI |
| `target/release/listen` | CLI pipeline |
| `target/release/ort-probe` | Print ORT execution providers |

`ort` is pinned to `=2.0.0-rc.10` (Rust 1.85; newer RCs need 1.88+). GUI uses `eframe`/`egui` **0.31**.

## Architecture

```
AudioSource (PipeWire mic | monitor | fake | WAV file)
    → optional VAD node (Silero ONNX or energy fallback)
    → ASR node (Whisper ggml by default — real transcription)
    → optional dual-decode / MT  → Original + English (GUI)
    → optional TTS node → PipeWire playback (beep stub or Piper session)
```

| Layer | Role |
|--------|------|
| `audio/` | PCM bus (**f32 mono @ 16 kHz**), resample, PipeWire capture + playback, fake + WAV |
| `runtime/` | `ort` sessions, EP probe & priority chain |
| `nodes/` | Capability traits, registry, VAD/ASR/MT/TTS plugs |
| `profile/` | Profile + manifest JSON |
| `pipeline/` | Source → nodes → stdout / `PipelineEvent` / TTS |
| `bin/edge_ort_gui.rs` | eframe/egui desktop UI |

### Execution providers

On ORT session load the runtime probes and prefers (degrading gracefully):

1. TensorRT → 2. CUDA → 3. ROCm → 4. OpenVINO → 5. CPU

Manifests may override `ep_prefs`. Use `ort-probe` to print what this build can see.

ASR with `meta.backend = "whisper_ggml"` uses **whisper.cpp** (not ORT EPs). VAD and TTS remain on ORT.

### Adding a model (manifest)

1. Place weights under `models/` (see [`models/README.md`](models/README.md) and `scripts/download-models.sh`).
2. Copy an example under `profiles/default/` and set `id`, `capability`, `model_path`, tensor names, `sample_rate`, `languages`, `ep_prefs`, `meta.backend`.
3. Point `profiles/<name>/profile.json` → that manifest for the capability key.
4. Restart `listen` / GUI with the profile name.

## Download models

```bash
./scripts/download-models.sh
# Default WHISPER_SIZES="tiny base"  (DOWNLOAD_MAX_MB default 600)

WHISPER_SIZES="tiny base small" ./scripts/download-models.sh
WHISPER_SIZES="tiny base small medium" DOWNLOAD_MAX_MB=2000 ./scripts/download-models.sh

# Optional Xenova Whisper ONNX (experimental ORT path):
FETCH_WHISPER_ONNX=1 ./scripts/download-models.sh
```

| Artifact | Required? | Notes |
|----------|-----------|--------|
| `ggml-tiny.bin` | default ASR | ~75 MB multilingual |
| `ggml-base.bin` | recommended | ~142 MB; GUI picker |
| `ggml-small.bin` / `medium` / `large-v3` | optional | GUI picker; fetch via `WHISPER_SIZES` |
| `silero_vad.onnx` | real VAD | ~2 MB |
| `jfk.wav` | smoke fixture | Also under `tests/fixtures/` |
| `tiny_asr_energy.onnx` | demo ORT ASR | Bundled |
| `en_US-lessac-low.onnx` (+ `.onnx.json`) | optional TTS | Piper |
| `nllb.onnx` | optional text MT | Skipped; passthrough until present |

Large `*.bin` / `*.onnx` / `*.wav` under `models/` are gitignored (except the bundled energy demo).

## Whisper ASR (default)

Default profile: `asr.whisper.json` → `models/ggml-tiny.bin`, `meta.backend = "whisper_ggml"`.

The GUI can switch sizes at Start via **Whisper model** (tiny / base / small / medium / large-v3) by calling `AsrNode::set_model_path`. Missing files show `✗ missing` and Start errors with a download hint.

### Languages

GUI picker (Whisper codes):

| UI | Code | Notes |
|----|------|--------|
| auto | `auto` | Detect |
| EN FR ES AR | `en` `fr` `es` `ar` | |
| RU HI KO JA | `ru` `hi` `ko` `ja` | |
| ZH / ZH-TW | `zh` | Whisper has one Chinese code; ZH-TW still uses `zh` |
| UR | `ur` | Urdu (Pakistani) |
| FA | `fa` | Farsi / Iranian Persian |
| TR AZ HY | `tr` `az` `hy` | Turkish, Azerbaijani, Armenian |

Or set `meta.language` in the manifest. `meta.task`: `transcribe` or `translate` (speech → English).

### Translate → EN (side-by-side)

With **Translate → EN** enabled (default in the GUI), the pipeline dual-decodes Whisper:

1. **Original** — `transcribe` (same language as picker / auto)
2. **English** — Whisper `translate` task (speech → English)

A separate text MT model (`nllb.onnx`) is still a passthrough stub; non-English → English for the right panel does **not** need it.

`conf=` in the transcript is the **mean of whisper.cpp per-token probabilities** (`full_get_token_prob`), not a hardcoded constant.

### ORT Whisper (experimental)

`meta.backend = "whisper"` loads a single-file ONNX if present. Full mel → encoder → decoder → tokenizer is not complete; prefer `whisper_ggml` for readable text.

## GUI

```bash
# Preferred helper (PipeWire + XDG_RUNTIME_DIR + DISPLAY):
./scripts/run-gui.sh

# Or from crate root so profiles/ and models/ resolve:
./target/release/edge-ort-gui
cargo run --release --bin edge-ort-gui
```

### Features

- **Source:** fake / mic / monitor (monitor needs GUI and Chrome/apps on the **same** PipeWire/`XDG_RUNTIME_DIR`)
- **Language** picker (see table above)
- **Whisper model** size picker + present/missing status
- **Translate → EN**, VAD, TTS toggles
- **Side-by-side** Original | English panels
- EP status, VAD readout, Start / Stop / Clear
- Unicode fonts: DejaVu, Harmattan (Arabic), Martel (Devanagari), Mirza, optional Noto CJK under `assets/fonts/`

### PipeWire tips

- Fake source works without PipeWire.
- Mic/monitor need a live PipeWire session (`pipewire`, `wireplumber`, often `pipewire-pulse`).
- If system audio from a browser is silent in the GUI, align `XDG_RUNTIME_DIR` (see `scripts/run-gui.sh`).

### CJK fonts (optional)

egui needs **TTF**, not TTC. Extract once:

```bash
./scripts/extract-cjk-fonts.sh
# writes assets/fonts/NotoSansCJK{sc,tc,jp,kr}-Regular.ttf (~16 MB each)
# those TTFs are gitignored; see assets/fonts/README.md
```

Without them, Latin/Cyrillic/Arabic/Devanagari still work; Chinese/Japanese/Korean glyphs may tofu.

## CLI

```bash
cargo run --release --bin ort-probe

# Whisper on sample WAV
cargo run --release --bin listen -- \
  --source file --file models/jfk.wav --no-vad --no-mt

# Fake sine (no words expected)
cargo run --release --bin listen -- --source fake --profile default --tts off

# Mic / sink monitor
cargo run --release --bin listen -- --source mic --profile default
cargo run --release --bin listen -- --source monitor --profile default --tts on
```

Useful flags: `--no-vad`, `--no-mt`, `--max-chunks N`, `--fake-secs 2`, `--file PATH`.

## Piper / TTS playback

- If Piper onnx+json are present, the TTS node loads the ORT session; phoneme frontend is not fully wired yet, so audio may still be a length-scaled beep.
- PipeWire playback streams PCM to the default sink when TTS is on. If PipeWire is down, playback logs RMS and continues.

## Layout

```
src/
  audio/       pcm, resample, pipewire, fake, wav
  runtime/     ort sessions + EP chain
  nodes/       vad, asr, mt, tts, registry
  pipeline/    runner + PipelineEvent
  profile/     profile + manifest types
  bin/listen.rs | ort-probe.rs | edge_ort_gui.rs
profiles/default/
models/                  weights (mostly gitignored)
assets/fonts/            optional CJK TTFs + README
scripts/download-models.sh
scripts/run-gui.sh
scripts/extract-cjk-fonts.sh
tests/fixtures/jfk.wav
```

## Tests

```bash
./scripts/download-models.sh
cargo test
```

`tests/whisper_asr_smoke.rs` runs Whisper on `jfk.wav` and asserts recognizable English words.

## Notes

- **PipeWire daemon**: capture/playback need a live session. Headless: `--source fake` or `--source file`.
- **ort pin**: keep `=2.0.0-rc.10` unless you bump Rust past 1.85.
- **cmake**: required to build `whisper-rs` / whisper.cpp.
- **Text MT (NLLB)**: still stub; English panel uses Whisper’s speech→English translate path.
