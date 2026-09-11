# edge-ort-runtime

[![CI](https://github.com/dkarapetyan-afk/edge-ort-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/dkarapetyan-afk/edge-ort-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Android Application (Jetpack Compose & Mobile Runtime)

In addition to the desktop Linux pipeline, `edge-ort-runtime` includes a full-featured native Android application targeting **Android SDK 36 (Android 16)**:

* **Material 3 UI ([`MainActivity.kt`](bindings/android/src/main/kotlin/com/edgeort/android/MainActivity.kt) & [`MainSpeechScreen.kt`](bindings/android/src/main/kotlin/com/edgeort/android/ui/MainSpeechScreen.kt))**: Live animated speech probability meter, dual side-by-side transcript cards with automatic RTL text direction (Arabic, Urdu, Farsi), multilingual language selector, and active NPU/GPU execution provider badges.
* **Persistent Foreground Service ([`SpeechRecognitionService.kt`](bindings/android/src/main/kotlin/com/edgeort/android/SpeechRecognitionService.kt))**: Runs with `FOREGROUND_SERVICE_TYPE_MICROPHONE`, `WakeLock`, and `OnAudioFocusChangeListener` to capture audio and perform on-device inference even when the screen is locked.
* **Floating Live Caption Overlay ([`FloatingSubtitleOverlayService.kt`](bindings/android/src/main/kotlin/com/edgeort/android/ui/FloatingSubtitleOverlayService.kt))**: System-wide draggable translucent live subtitles displayed over third-party apps (YouTube, Zoom, phone calls).
* **Hardware Acceleration**: Probes Qualcomm Hexagon NPU (`QNN`), Android `NNAPI`, and ARM NEON SIMD (`XNNPACK`).
* **Multi-Source Audio Capture (Speakers & Mic)**: Supports real-time internal playback audio capture via `AudioPlaybackCaptureConfiguration` (Android 10+ / API 29+) with soft-clipping digital mixing (`AudioRecordCapture.kt`). Allows transcribing both speaker playback (YouTube, media players, calls, games) and spoken microphone audio simultaneously, with selectable modes:
  - `All Audio (Speakers + Mic)`: Combines system playback and microphone with real-time saturation-clamped PCM mixing.
  - `Speakers / Playback Only`: Transcribes pure device internal media audio without background microphone noise.
  - `Microphone Only`: Direct ambient microphone capture.
* **Background Model Downloader ([`ModelDownloadManager.kt`](bindings/android/src/main/kotlin/com/edgeort/android/download/ModelDownloadManager.kt))**: Jetpack `WorkManager` streaming worker with Wi-Fi constraints, SHA-256 validation, and atomic commits for Silero VAD, Whisper INT8, Opus-MT, and SmolLM SLMs.
* **Local SLM Cleaner ([`slm.rs`](src/nodes/slm.rs))**: Disfluency and conversational filler removal (`um`, `uh`, `like`, `you know`), capitalization fixing, and action item bullet extraction.
* **16KB Page Alignment**: Fully compliant with Android 15+ 16KB page size kernel requirements (Pixel 9 / Pixel 10 series, Android 16/17 SDK 36) using ONNX Runtime 1.22.0+ with 16KB-page-aligned native ELF libraries (`-Wl,-z,max-page-size=16384`, `useLegacyPackaging = false`).

### Building the Android APK

```bash
# Set Android SDK path if not already in environment
export ANDROID_HOME=~/Android/Sdk

# Build Debug APK and run unit tests
./gradlew assembleDebug test

# Output APK:
# bindings/android/build/outputs/apk/debug/edge-ort-android-debug.apk
```

### Running Android Unit Tests

```bash
./gradlew testDebugUnitTest
```

### Deploying to an Android Device (WSL / Linux)

1. **Enable Developer Options & USB Debugging** on your phone (or start an Android Virtual Device emulator).
2. **List Connected Devices**:
   ```bash
   adb devices
   ```
   *WSL Tip*: If running in WSL and accessing Windows ADB server or USB devices:
   ```bash
   # If running alongside Windows ADB daemon:
   powershell.exe -Command "& 'C:\Users\<User>\AppData\Local\Android\Sdk\platform-tools\adb.exe' -P 5038 devices"
   # Or attach USB to WSL directly:
   usbipd wsl attach --busid <BUSID>
   ```
3. **Install and Launch on Device**:
   ```bash
   adb -s <DEVICE_ID> install -r bindings/android/build/outputs/apk/debug/edge-ort-android-debug.apk
   adb -s <DEVICE_ID> shell am start -n com.edgeort.android/.MainActivity
   ```

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
- **ort pin**: currently `=2.0.0-rc.13` (see `Cargo.toml`); MSRV follows that crate.
- **cmake**: required to build `whisper-rs` / whisper.cpp.
- **Text MT (NLLB)**: still stub; English panel uses Whisper’s speech→English translate path.

## Related projects (comparison)

Based on the architecture of **edge-ort-runtime**—a native **Rust** edge audio pipeline combining **PipeWire / system audio capture**, **ONNX Runtime (`ort`)**, **Whisper / GGML ASR**, **Silero VAD**, **machine translation (MT)**, and **TTS (Piper)** with both CLI and GUI—here are similar open-source projects by focus.

### Most architecturally similar (multi-modal speech + ONNX / Rust / edge)

* **[sherpa-onnx (k2-fsa)](https://github.com/k2-fsa/sherpa-onnx)** — Complete cross-platform speech toolkit on ONNX Runtime: offline/streaming ASR (Whisper, Moonshine, SenseVoice, Zipformer, Paraformer), Silero VAD, TTS (Piper, VITS, Kokoro), spoken language ID. Native C++, Rust, Go, Python, C# APIs for edge devices (embedded Linux, Pi, x86/ARM, desktop). Closest full-stack parallel.
* **[Murmure](https://github.com/Kieirra/murmure)** — Fully local, private speech-to-text app in Rust using NVIDIA Parakeet and ONNX Runtime (`ort`).
* **[Candle Whisper (Hugging Face Candle)](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)** — Pure-Rust ML framework for serverless/edge; zero-Python Whisper with CPU, CUDA, Metal, and WebAssembly.

### Live system audio & desktop GUI (real-time transcription / translation)

* **[Buzz](https://github.com/chidiwilliams/buzz)** — Desktop app for real-time transcription/translation from mic or system audio; live captions, file transcription, dual-track translation; backends include `whisper.cpp`, `faster-whisper`, OpenAI Whisper.
* **[LiveCaptions (Linux)](https://github.com/abb128/LiveCaptions)** — Native Linux live captions from internal audio (PipeWire / PulseAudio) or mics; accessibility-focused offline captioning.
* **[Whispering](https://github.com/braden-w/whispering)** — Cross-platform desktop / hotkey tool for offline voice-to-text and live translation; `whisper.cpp` / ONNX, local capture, speech→English.

### Real-time streaming & client–server pipelines

* **[WhisperLive (Collabora)](https://github.com/collabora/WhisperLive)** — Near-real-time Whisper streaming server/client over WebSockets; PCM capture, VAD segmentation, live transcripts/translations; `faster-whisper` + TensorRT.
* **[VoiceStreamAI](https://github.com/alesaccoia/VoiceStreamAI)** — Lightweight low-latency pipeline: chunking, VAD, Whisper ASR, and LLM/MT (see also KoljaB's [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)).

### High-performance core engines (backends / plugs)

Useful as drop-in ideas for this repo’s capability plugs:

| Project | Capabilities | Language | Highlights |
| --- | --- | --- | --- |
| **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** | ASR / speech→EN | C/C++ (`whisper-rs`) | Low-memory GGML backend used inside `edge-ort-runtime` |
| **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** | ASR / MT | Python / CTranslate2 | Often much faster than stock Whisper with INT8 |
| **[Piper](https://github.com/rhasspy/piper)** | TTS | C++ / ONNX Runtime | Fast local neural TTS for edge |
| **[Silero VAD](https://github.com/snakers4/silero-vad)** | VAD | ONNX / TorchScript | Small (~2 MB) speech activity detector |

### Summary

* **Closest complete alternative (ASR + TTS + VAD on ONNX, multi-platform):** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
* **Ready desktop UI for live mic/system STT + translation:** [Buzz](https://github.com/chidiwilliams/buzz) or [LiveCaptions](https://github.com/abb128/LiveCaptions)
* **Pure Rust embedding:** [Candle](https://github.com/huggingface/candle), [whisper-rs](https://github.com/tazz4843/whisper-rs), and [ort](https://github.com/pykeio/ort)
