//! TTS node — Piper plug when model present; beep stub otherwise.
//! Playback goes through PipeWire when available (see `play_pcm`).

use super::{NodeError, TtsOut};
use crate::audio::{play_pcm, playback_log, PCM_SAMPLE_RATE};
use crate::profile::Manifest;
use crate::runtime::SessionManager;
use std::f32::consts::PI;
use std::path::PathBuf;
use tracing::{info, warn};

pub struct TtsNode {
    pub manifest: Manifest,
    pub model_ready: bool,
    session_key: String,
    loaded: bool,
    config_path: Option<PathBuf>,
}

impl TtsNode {
    pub fn from_manifest(manifest: Manifest) -> Self {
        let model_ready = manifest.model_path.exists();
        if !model_ready {
            warn!(
                id = %manifest.id,
                path = %manifest.model_path.display(),
                "TTS model missing — beep stub"
            );
        }
        // Piper convention: voice.onnx.json next to voice.onnx
        let config_path = {
            let p = PathBuf::from(format!("{}.json", manifest.model_path.display()));
            if p.exists() {
                Some(p)
            } else {
                None
            }
        };
        if model_ready && config_path.is_none() {
            warn!(
                id = %manifest.id,
                "TTS ONNX present but {}.json config missing — synthesis will stub",
                manifest.model_path.display()
            );
        }
        Self {
            session_key: format!("tts:{}", manifest.id),
            manifest,
            model_ready,
            loaded: false,
            config_path,
        }
    }

    pub fn ensure_loaded(&mut self, sessions: &mut SessionManager) -> Result<(), NodeError> {
        if self.loaded || !self.model_ready {
            return Ok(());
        }
        // Piper phoneme→audio needs espeak-ng + phoneme ids; we load the session to
        // prove the plug wires, but full synthesis remains documented as TODO unless
        // a simpler PCM-producing ONNX is provided.
        match sessions.load(
            &self.session_key,
            &self.manifest.model_path,
            &self.manifest.ep_prefs,
        ) {
            Ok(_) => {
                self.loaded = true;
                info!(
                    id = %self.manifest.id,
                    has_config = self.config_path.is_some(),
                    "TTS ONNX session ready (phoneme frontend may still be stubbed)"
                );
            }
            Err(e) => {
                warn!(error = %e, "TTS ONNX load failed — beep stub");
            }
        }
        Ok(())
    }

    /// Produce PCM. Real Piper needs phoneme IDs; until then emit a beep whose
    /// length tracks text. If the ONNX session loaded, note it in logs.
    pub fn process(
        &mut self,
        sessions: &mut SessionManager,
        text: &str,
    ) -> Result<TtsOut, NodeError> {
        let _ = self.ensure_loaded(sessions);
        if text.is_empty() {
            return Ok(TtsOut {
                pcm: Vec::new(),
                sample_rate: self.manifest.sample_rate.max(PCM_SAMPLE_RATE),
            });
        }

        if self.loaded {
            info!(
                id = %self.manifest.id,
                chars = text.len(),
                "TTS: Piper session loaded; phoneme→audio frontend not yet wired — beep stub"
            );
        }

        let sr = if self.manifest.sample_rate > 0 {
            self.manifest.sample_rate
        } else {
            PCM_SAMPLE_RATE
        };
        let ms = (50 + text.len() * 15).min(800) as u32;
        let n = (sr * ms / 1000) as usize;
        let mut pcm = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let env = if i < 64 {
                i as f32 / 64.0
            } else if i + 64 > n {
                (n - i) as f32 / 64.0
            } else {
                1.0
            };
            pcm.push((2.0 * PI * 880.0 * t).sin() * 0.15 * env);
        }
        Ok(TtsOut {
            pcm,
            sample_rate: sr,
        })
    }
}

/// Play PCM via PipeWire when possible; always logs RMS as a fallback.
pub fn playback_stub(pcm: &[f32], sample_rate: u32) {
    if pcm.is_empty() {
        return;
    }
    if let Err(e) = play_pcm(pcm, sample_rate) {
        warn!(error = %e, "playback error");
        playback_log(pcm, sample_rate);
    }
}
