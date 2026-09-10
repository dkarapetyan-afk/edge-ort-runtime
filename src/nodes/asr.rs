//! ASR node — pluggable backends via manifest `meta.backend`:
//!
//! - `tiny_energy` — bundled demo ONNX (energy → 4 logits / labels)
//! - `whisper` — ORT Whisper encoder path (partial; diagnostic stub until full decode)
//! - `whisper_ggml` — **real** Whisper via whisper.cpp (`whisper-rs` + `ggml-*.bin`)
//!
//! Default profile uses `whisper_ggml` so spoken English yields readable text.
//! VAD / TTS / EP probing remain on ONNX Runtime.

use super::{AsrOut, NodeError};
use crate::profile::Manifest;
use crate::runtime::SessionManager;
use ndarray::Array2;
use ort::value::Tensor;
use std::sync::Arc;
use tracing::{debug, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const LABELS: [&str; 4] = ["[silence]", "[noise]", "[speech]", "[utterance]"];

/// Minimum buffered PCM before running ggml Whisper (2 s @ 16 kHz).
/// whisper.cpp rejects inputs shorter than 1000 ms; 800 ms chunks produced empty transcripts.
const WHISPER_MIN_SAMPLES: usize = 16_000 * 2;
/// Absolute floor for whisper_full (pad with silence if somehow shorter).
const WHISPER_HARD_MIN_SAMPLES: usize = 16_000;
/// Soft cap so streaming chunks don't grow without bound (≈30 s Whisper window).
const WHISPER_MAX_SAMPLES: usize = 16_000 * 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsrBackend {
    TinyEnergy,
    WhisperOrt,
    WhisperGgml,
}

pub struct AsrNode {
    pub manifest: Manifest,
    session_key: String,
    loaded: bool,
    backend: AsrBackend,
    /// whisper.cpp context (shared; state created per decode).
    whisper_ctx: Option<Arc<WhisperContext>>,
    /// Streaming PCM accumulator for short chunks (mic / fake).
    pcm_buf: Vec<f32>,
    language: String,
    translate: bool,
    /// When true, whisper_ggml decodes twice: original (transcribe) + English (translate).
    dual_translate: bool,
    n_threads: i32,
}

impl AsrNode {
    pub fn from_manifest(manifest: Manifest) -> Self {
        let backend = match manifest
            .meta
            .get("backend")
            .and_then(|v| v.as_str())
            .unwrap_or("")
        {
            "whisper_ggml" | "whisper-cpp" | "ggml" => AsrBackend::WhisperGgml,
            "whisper" | "whisper_ort" => AsrBackend::WhisperOrt,
            "tiny_energy" => AsrBackend::TinyEnergy,
            _ if manifest.id.contains("whisper")
                && manifest
                    .model_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == "bin")
                    .unwrap_or(false) =>
            {
                AsrBackend::WhisperGgml
            }
            _ if manifest.id.contains("whisper") => AsrBackend::WhisperOrt,
            _ => AsrBackend::TinyEnergy,
        };

        let language = manifest
            .meta
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                manifest
                    .languages
                    .iter()
                    .find(|l| *l != "multilingual" && *l != "und")
                    .cloned()
            })
            .unwrap_or_else(|| "en".into());

        let translate = manifest
            .meta
            .get("task")
            .and_then(|v| v.as_str())
            .map(|t| t.eq_ignore_ascii_case("translate"))
            .unwrap_or(false);

        let n_threads = manifest
            .meta
            .get("n_threads")
            .and_then(|v| v.as_i64())
            .map(|n| n as i32)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(4) as i32)
                    .unwrap_or(2)
            });

        let session_key = format!("asr:{}", manifest.id);
        Self {
            manifest,
            session_key,
            loaded: false,
            backend,
            whisper_ctx: None,
            pcm_buf: Vec::new(),
            language,
            translate,
            dual_translate: false,
            n_threads,
        }
    }

    /// Override recognition language at runtime (GUI / CLI). Codes: en, fr, es, ar, auto, …
    pub fn set_language(&mut self, language: impl Into<String>) {
        self.language = language.into();
        // Clear streaming buffer so a language switch doesn't mix windows.
        self.pcm_buf.clear();
    }

    /// Point at a different ggml (or ONNX) weights file; next `ensure_loaded` reloads.
    pub fn set_model_path(&mut self, path: impl Into<std::path::PathBuf>) {
        self.manifest.model_path = path.into();
        self.whisper_ctx = None;
        self.loaded = false;
        self.pcm_buf.clear();
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn set_translate(&mut self, translate: bool) {
        self.translate = translate;
    }

    pub fn set_dual_translate(&mut self, dual: bool) {
        self.dual_translate = dual;
    }

    pub fn dual_translate(&self) -> bool {
        self.dual_translate
    }

    pub fn ensure_loaded(&mut self, sessions: &mut SessionManager) -> Result<(), NodeError> {
        if self.loaded {
            return Ok(());
        }

        match self.backend {
            AsrBackend::WhisperGgml => {
                if !self.manifest.model_path.exists() {
                    warn!(
                        id = %self.manifest.id,
                        path = %self.manifest.model_path.display(),
                        "Whisper ggml model missing — stub transcripts until download"
                    );
                    return Ok(());
                }
                let ctx = WhisperContext::new_with_params(
                    self.manifest
                        .model_path
                        .to_str()
                        .ok_or_else(|| NodeError::Other("non-utf8 model path".into()))?,
                    WhisperContextParameters::default(),
                )
                .map_err(|e| NodeError::Session(format!("whisper.cpp load: {e}")))?;
                self.whisper_ctx = Some(Arc::new(ctx));
                self.loaded = true;
                info!(
                    id = %self.manifest.id,
                    backend = "whisper_ggml",
                    lang = %self.language,
                    path = %self.manifest.model_path.display(),
                    "ASR whisper.cpp ready"
                );
                Ok(())
            }
            AsrBackend::TinyEnergy | AsrBackend::WhisperOrt => {
                if !self.manifest.model_path.exists() {
                    warn!(
                        id = %self.manifest.id,
                        path = %self.manifest.model_path.display(),
                        "ASR model missing — will emit stub transcripts"
                    );
                    return Ok(());
                }
                sessions
                    .load(
                        &self.session_key,
                        &self.manifest.model_path,
                        &self.manifest.ep_prefs,
                    )
                    .map_err(|e| NodeError::Session(e.to_string()))?;
                self.loaded = true;
                info!(
                    id = %self.manifest.id,
                    backend = ?self.backend,
                    "ASR ONNX session ready"
                );
                Ok(())
            }
        }
    }

    pub fn process(
        &mut self,
        sessions: &mut SessionManager,
        pcm: &[f32],
    ) -> Result<AsrOut, NodeError> {
        self.ensure_loaded(sessions)?;

        if !self.loaded {
            return Ok(stub_asr(pcm));
        }

        match self.backend {
            AsrBackend::TinyEnergy => self.process_tiny_energy(sessions, pcm),
            AsrBackend::WhisperOrt => self.process_whisper_ort(sessions, pcm),
            AsrBackend::WhisperGgml => self.process_whisper_ggml(pcm),
        }
    }

    fn process_tiny_energy(
        &mut self,
        sessions: &mut SessionManager,
        pcm: &[f32],
    ) -> Result<AsrOut, NodeError> {
        let session = sessions
            .get_mut(&self.session_key)
            .ok_or_else(|| NodeError::NotReady("asr session".into()))?;

        let input_name = self
            .manifest
            .inputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("audio");

        let n = pcm.len().max(1);
        let mut row = pcm.to_vec();
        if row.is_empty() {
            row.push(0.0);
        }
        let arr = Array2::from_shape_vec((1, n), row)
            .map_err(|e| NodeError::Other(e.to_string()))?;

        let input = Tensor::from_array(arr).map_err(|e| NodeError::Session(e.to_string()))?;

        let outputs = session
            .run(ort::inputs![input_name => input])
            .map_err(|e| NodeError::Session(e.to_string()))?;

        let out_name = self
            .manifest
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("logits");

        let logits_val = outputs
            .get(out_name)
            .ok_or_else(|| NodeError::Other(format!("missing output tensor {out_name}")))?;

        let (_shape, data) = logits_val
            .try_extract_tensor::<f32>()
            .map_err(|e| NodeError::Session(e.to_string()))?;

        let (idx, conf) = argmax(data);
        let label = LABELS.get(idx).copied().unwrap_or("[?]");
        debug!(label, conf, "asr logits");

        let text = if label == "[silence]" || label == "[noise]" {
            String::new()
        } else {
            label.to_string()
        };

        Ok(AsrOut {
            text,
            translation: None,
            confidence: conf,
        })
    }

    /// ORT Whisper plug — session loads when a single-file ONNX is present.
    /// Full encoder→decoder + tokenizer remains TODO; prefer `whisper_ggml`.
    fn process_whisper_ort(
        &mut self,
        sessions: &mut SessionManager,
        pcm: &[f32],
    ) -> Result<AsrOut, NodeError> {
        let _session = sessions
            .get_mut(&self.session_key)
            .ok_or_else(|| NodeError::NotReady("asr whisper session".into()))?;

        let energy: f32 = if pcm.is_empty() {
            0.0
        } else {
            pcm.iter().map(|x| x.abs()).sum::<f32>() / pcm.len() as f32
        };
        if energy < 0.01 {
            return Ok(AsrOut {
                text: String::new(),
                translation: None,
                confidence: 0.0,
            });
        }

        Ok(AsrOut {
            text: format!(
                "[whisper-ort plug loaded; use backend=whisper_ggml for real decode energy={energy:.4}]"
            ),
            translation: None,
            confidence: energy.min(1.0),
        })
    }

    fn process_whisper_ggml(&mut self, pcm: &[f32]) -> Result<AsrOut, NodeError> {
        if pcm.is_empty() {
            return Ok(AsrOut {
                text: String::new(),
                translation: None,
                confidence: 0.0,
            });
        }

        // Large pulls (file source / long utterance): decode immediately.
        if pcm.len() >= WHISPER_MIN_SAMPLES {
            // If we have leftover buffer, prepend then clear.
            if !self.pcm_buf.is_empty() {
                let mut combined = std::mem::take(&mut self.pcm_buf);
                combined.extend_from_slice(pcm);
                if combined.len() > WHISPER_MAX_SAMPLES {
                    let skip = combined.len() - WHISPER_MAX_SAMPLES;
                    combined.drain(..skip);
                }
                return self.decode_whisper_pcm(&combined);
            }
            let slice = if pcm.len() > WHISPER_MAX_SAMPLES {
                &pcm[pcm.len() - WHISPER_MAX_SAMPLES..]
            } else {
                pcm
            };
            return self.decode_whisper_pcm(slice);
        }

        // Short streaming chunks: accumulate.
        self.pcm_buf.extend_from_slice(pcm);
        if self.pcm_buf.len() > WHISPER_MAX_SAMPLES {
            let skip = self.pcm_buf.len() - WHISPER_MAX_SAMPLES;
            self.pcm_buf.drain(..skip);
        }
        if self.pcm_buf.len() < WHISPER_MIN_SAMPLES {
            return Ok(AsrOut {
                text: String::new(),
                translation: None,
                confidence: 0.0,
            });
        }
        let buffered = std::mem::take(&mut self.pcm_buf);
        self.decode_whisper_pcm(&buffered)
    }

    /// Single or dual decode: always produce original text; optionally English via Whisper translate.
    fn decode_whisper_pcm(&self, pcm: &[f32]) -> Result<AsrOut, NodeError> {
        // Dual path forces first pass as transcribe; single-mode uses self.translate.
        let first_translate = if self.dual_translate {
            false
        } else {
            self.translate
        };
        let mut out = self.run_whisper(pcm, first_translate)?;
        if out.text.is_empty() {
            return Ok(out);
        }

        if self.dual_translate {
            let lang = self.language.to_ascii_lowercase();
            if lang != "en" {
                let en = self.run_whisper(pcm, true)?;
                out.translation = if en.text.is_empty() {
                    None
                } else {
                    Some(en.text)
                };
            }
            // language == "en" → translation stays None (GUI shows "—")
        }
        Ok(out)
    }

    fn run_whisper(&self, pcm: &[f32], translate: bool) -> Result<AsrOut, NodeError> {
        let ctx = self
            .whisper_ctx
            .as_ref()
            .ok_or_else(|| NodeError::NotReady("whisper_ggml context".into()))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| NodeError::Session(format!("whisper state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(true);
        params.set_no_context(true);

        // Multilingual language selection: "auto" / "detect" → detect_language;
        // otherwise BCP-47 short code (en, fr, es, ar, …).
        let lang = self.language.to_ascii_lowercase();
        if lang == "auto" || lang == "detect" || lang == "multilingual" {
            params.set_detect_language(true);
            params.set_language(None);
        } else {
            params.set_detect_language(false);
            // Leak-free: FullParams needs &'a str living for the call — language String is on self.
            params.set_language(Some(self.language.as_str()));
        }

        // whisper.cpp requires ≥1000 ms of audio.
        let mut pcm_owned;
        let pcm: &[f32] = if pcm.len() < WHISPER_HARD_MIN_SAMPLES {
            pcm_owned = pcm.to_vec();
            pcm_owned.resize(WHISPER_HARD_MIN_SAMPLES, 0.0);
            &pcm_owned
        } else {
            pcm
        };

        state
            .full(params, pcm)
            .map_err(|e| NodeError::Session(format!("whisper full: {e}")))?;

        let n = state
            .full_n_segments()
            .map_err(|e| NodeError::Session(format!("whisper segments: {e}")))?;

        let mut text = String::new();
        let mut prob_sum = 0.0f32;
        let mut prob_n = 0usize;
        for i in 0..n {
            let seg = state
                .full_get_segment_text(i)
                .map_err(|e| NodeError::Session(format!("whisper text: {e}")))?;
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(seg.trim());

            // Average token probabilities from whisper.cpp (real confidence, not a stub).
            let n_tokens = state
                .full_n_tokens(i)
                .map_err(|e| NodeError::Session(format!("whisper n_tokens: {e}")))?;
            for t in 0..n_tokens {
                let p = state
                    .full_get_token_prob(i, t)
                    .map_err(|e| NodeError::Session(format!("whisper token_p: {e}")))?;
                if p.is_finite() && (0.0..=1.0).contains(&p) {
                    prob_sum += p;
                    prob_n += 1;
                }
            }
        }
        let text = sanitize_whisper_text(text);
        let confidence = if text.is_empty() {
            0.0
        } else if prob_n > 0 {
            (prob_sum / prob_n as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        debug!(
            len = text.len(),
            translate,
            confidence,
            tokens = prob_n,
            "whisper_ggml transcript"
        );

        Ok(AsrOut {
            text,
            translation: None,
            confidence,
        })
    }
}

fn stub_asr(pcm: &[f32]) -> AsrOut {
    let energy: f32 = if pcm.is_empty() {
        0.0
    } else {
        pcm.iter().map(|x| x.abs()).sum::<f32>() / pcm.len() as f32
    };
    let text = if energy < 0.01 {
        String::new()
    } else {
        format!("[stub-asr energy={energy:.4}]")
    };
    AsrOut {
        text,
        translation: None,
        confidence: energy.min(1.0),
    }
}

/// Drop empty / ellipsis-only Whisper hallucinations.
fn sanitize_whisper_text(text: String) -> String {
    let t = text.trim().to_string();
    if t.is_empty() {
        return String::new();
    }
    // Treat "..." / "…" / dots-only as empty.
    if t.chars().all(|c| c == '.' || c == '…' || c.is_whitespace()) {
        return String::new();
    }
    t
}

fn argmax(xs: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    let conf = (1.0 / (1.0 + (-best_v).exp())).clamp(0.0, 1.0);
    (best_i, conf)
}
