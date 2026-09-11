//! AudioSource → optional VAD → ASR → optional MT → stdout / events; optional TTS → playback.

use crate::audio::AudioSource;
use crate::nodes::playback_stub;
use crate::nodes::NodeRegistry;
use crate::runtime::SessionManager;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info};

pub struct PipelineConfig {
    pub enable_vad: bool,
    pub enable_mt: bool,
    pub enable_tts: bool,
    /// Stop after this many chunks (None = run until source ends / stop flag).
    pub max_chunks: Option<usize>,
    /// Cooperative cancellation (GUI Stop / Ctrl handlers).
    pub stop: Option<Arc<AtomicBool>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_vad: true,
            enable_mt: true,
            enable_tts: false,
            max_chunks: None,
            stop: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    Status(String),
    EpInfo { name: String, available: bool },
    Vad {
        speech_prob: f32,
        is_speech: bool,
    },
    Transcript {
        text: String,
        /// English when dual-decode / MT produced a translation.
        translation: Option<String>,
        confidence: f32,
    },
    Error(String),
    Done,
}

pub struct Pipeline {
    pub registry: NodeRegistry,
    pub sessions: SessionManager,
    pub config: PipelineConfig,
}

impl Pipeline {
    pub fn new(registry: NodeRegistry, config: PipelineConfig) -> Self {
        Self {
            registry,
            sessions: SessionManager::new(),
            config,
        }
    }

    /// Classic CLI path: print transcripts to stdout.
    pub fn run(&mut self, source: &mut dyn AudioSource) -> Result<()> {
        self.run_with_handler(source, |ev| {
            if let PipelineEvent::Transcript {
                text,
                translation,
                ..
            } = &ev
            {
                if let Some(tr) = translation {
                    println!("{text} => {tr}");
                } else {
                    println!("{text}");
                }
            }
            true
        })
    }

    /// Event-driven path for GUI / custom sinks. Handler returns `false` to stop.
    pub fn run_with_handler<F>(
        &mut self,
        source: &mut dyn AudioSource,
        mut handler: F,
    ) -> Result<()>
    where
        F: FnMut(PipelineEvent) -> bool,
    {
        info!(source = source.name(), "pipeline start");
        let _ = handler(PipelineEvent::Status(format!(
            "pipeline start ({})",
            source.name()
        )));

        // Whisper dual-decode → EN when MT/translate is enabled (MT node remains optional fallback).
        if self.config.enable_mt {
            if let Some(asr) = self.registry.asr.as_mut() {
                asr.set_dual_translate(true);
            }
        }

        let mut chunks = 0usize;

        loop {
            if self
                .config
                .stop
                .as_ref()
                .map(|s| s.load(Ordering::Relaxed))
                .unwrap_or(false)
            {
                info!("pipeline stop requested");
                break;
            }

            if let Some(max) = self.config.max_chunks {
                if chunks >= max {
                    break;
                }
            }

            let pcm = match source.pull() {
                Ok(p) => p,
                Err(e) => {
                    let msg = e.to_string();
                    let _ = handler(PipelineEvent::Error(msg.clone()));
                    return Err(e.into());
                }
            };
            if pcm.is_empty() {
                info!("audio source ended");
                break;
            }
            chunks += 1;

            // Optional VAD gate.
            if self.config.enable_vad {
                if let Some(vad) = self.registry.vad.as_mut() {
                    let out = vad.process(&mut self.sessions, &pcm)?;
                    debug!(prob = out.speech_prob, speech = out.is_speech, "vad");
                    if !handler(PipelineEvent::Vad {
                        speech_prob: out.speech_prob,
                        is_speech: out.is_speech,
                    }) {
                        break;
                    }
                    if !out.is_speech {
                        continue;
                    }
                }
            }

            let Some(asr) = self.registry.asr.as_mut() else {
                continue;
            };
            let asr_out = asr.process(&mut self.sessions, &pcm)?;
            if asr_out.text.is_empty() {
                continue;
            }

            let text = asr_out.text.clone();
            // Prefer Whisper dual-decode English; fall back to MT stub when absent.
            let translation = if let Some(tr) = asr_out.translation {
                Some(tr)
            } else if self.config.enable_mt {
                if let Some(mt) = self.registry.mt.as_mut() {
                    let mt_out = mt.process(&text)?;
                    if mt_out.text.is_empty() {
                        None
                    } else {
                        Some(mt_out.text)
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if !handler(PipelineEvent::Transcript {
                text: text.clone(),
                translation: translation.clone(),
                confidence: asr_out.confidence,
            }) {
                break;
            }

            if self.config.enable_tts {
                if let Some(tts) = self.registry.tts.as_mut() {
                    let speak = translation.as_deref().unwrap_or(text.as_str());
                    let tts_out = tts.process(&mut self.sessions, speak)?;
                    playback_stub(&tts_out.pcm, tts_out.sample_rate);
                }
            }
        }

        info!(chunks, "pipeline done");
        let _ = handler(PipelineEvent::Done);
        Ok(())
    }
}
