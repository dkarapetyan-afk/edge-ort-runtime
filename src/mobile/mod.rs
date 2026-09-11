//! Mobile embedding runtime for Android / iOS / embedded targets.
//!
//! Provides `MobilePipeline`, a high-level, thread-safe orchestrator designed for
//! integration with mobile audio capture (e.g. Oboe / AAudio on Android) and UI frameworks
//! (e.g. Jetpack Compose or Flutter).

pub mod uniffi_api;

pub use uniffi_api::{
    get_execution_providers, get_supported_languages, LanguageInfo, NativePipelineError,
    NativePipelineHandle, NativeSpeechListener, ProviderInfoRecord,
};

use crate::audio::PushAudioSource;
use crate::nodes::NodeRegistry;
use crate::pipeline::{Pipeline, PipelineConfig, PipelineEvent};
use crate::profile::{Profile, SupportedLanguage};
use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct MobileConfig {
    pub language: SupportedLanguage,
    pub enable_vad: bool,
    pub enable_mt: bool,
    pub profile_path: Option<PathBuf>,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            language: SupportedLanguage::Auto,
            enable_vad: true,
            enable_mt: true,
            profile_path: None,
        }
    }
}

pub struct MobilePipeline {
    config: MobileConfig,
    audio_source: PushAudioSource,
    stop_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<()>>>,
}

impl MobilePipeline {
    pub fn new(config: MobileConfig) -> Self {
        Self {
            config,
            audio_source: PushAudioSource::new("mobile-mic"),
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker: None,
        }
    }

    pub fn config(&self) -> &MobileConfig {
        &self.config
    }

    pub fn audio_source(&self) -> PushAudioSource {
        self.audio_source.clone()
    }

    /// Push mono f32 @ 16 kHz samples directly into the input queue.
    pub fn push_samples(&self, samples: &[f32]) {
        self.audio_source.push(samples);
    }

    /// Push 16-bit signed PCM samples (normalized to [-1.0, 1.0]).
    pub fn push_pcm16(&self, pcm16: &[i16]) {
        self.audio_source.push_pcm16(pcm16);
    }

    /// Push multi-channel f32 samples at any sample rate; downmixes to mono and resamples to 16 kHz.
    pub fn push_resampled(&self, samples: &[f32], sample_rate: u32, channels: u16) {
        self.audio_source.push_resampled(samples, sample_rate, channels);
    }

    /// Push multi-channel 16-bit PCM at any sample rate; downmixes and resamples to 16 kHz.
    pub fn push_pcm16_resampled(&self, pcm16: &[i16], sample_rate: u32, channels: u16) {
        self.audio_source.push_pcm16_resampled(pcm16, sample_rate, channels);
    }

    pub fn is_running(&self) -> bool {
        self.worker.as_ref().map(|w| !w.is_finished()).unwrap_or(false)
    }

    /// Start pipeline worker thread. Events are delivered to `listener`.
    pub fn start<F>(&mut self, mut listener: F) -> Result<()>
    where
        F: FnMut(PipelineEvent) -> bool + Send + 'static,
    {
        if self.is_running() {
            bail!("mobile pipeline is already running");
        }

        self.stop_flag.store(false, Ordering::SeqCst);
        let stop_flag = Arc::clone(&self.stop_flag);
        let mut source = self.audio_source.clone();

        // Build registry from profile or default fallback
        let registry = if let Some(ref path) = self.config.profile_path {
            if path.exists() {
                let profile = Profile::load(path).context("load profile")?;
                let manifests = profile.load_manifests(path).context("load manifests")?;
                let mut reg = NodeRegistry::from_manifests(manifests);
                if let Some(asr) = reg.asr.as_mut() {
                    asr.set_language(self.config.language.code());
                }
                reg
            } else {
                warn!(path = %path.display(), "profile path not found, using empty registry");
                NodeRegistry::default()
            }
        } else {
            NodeRegistry::default()
        };

        let pipeline_config = PipelineConfig {
            enable_vad: self.config.enable_vad,
            enable_mt: self.config.enable_mt,
            enable_tts: false,
            max_chunks: None,
            stop: Some(Arc::clone(&stop_flag)),
        };

        let mut pipeline = Pipeline::new(registry, pipeline_config);

        let handle = thread::Builder::new()
            .name("mobile-pipeline".into())
            .spawn(move || {
                info!("mobile pipeline background worker started");
                let res = pipeline.run_with_handler(&mut source, |ev| {
                    let keep = listener(ev);
                    keep && !stop_flag.load(Ordering::Relaxed)
                });
                info!("mobile pipeline background worker finished");
                res
            })
            .context("spawn mobile pipeline worker")?;

        self.worker = Some(handle);
        Ok(())
    }

    /// Stop pipeline and wait for background thread to exit.
    pub fn stop(&mut self) -> Result<()> {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.audio_source.close();
        if let Some(handle) = self.worker.take() {
            match handle.join() {
                Ok(res) => res?,
                Err(_) => bail!("mobile pipeline worker thread panicked"),
            }
        }
        Ok(())
    }
}

impl Drop for MobilePipeline {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_mobile_pipeline_lifecycle() {
        let config = MobileConfig {
            language: SupportedLanguage::En,
            enable_vad: true,
            enable_mt: true,
            profile_path: None,
        };

        let mut pipeline = MobilePipeline::new(config);
        assert!(!pipeline.is_running());
        assert_eq!(pipeline.config().language, SupportedLanguage::En);

        let (tx, rx) = mpsc::channel();
        pipeline
            .start(move |ev| {
                let _ = tx.send(ev);
                true
            })
            .unwrap();

        assert!(pipeline.is_running());

        // First event should be Status indicating start
        let ev = rx.recv_timeout(Duration::from_millis(500)).unwrap();
        match ev {
            PipelineEvent::Status(s) => assert!(s.contains("pipeline start")),
            other => panic!("expected Status event, got: {:?}", other),
        }

        // Push some 16 kHz silence
        pipeline.push_samples(&[0.0; 3200]);

        // Stop pipeline cleanly
        pipeline.stop().unwrap();
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_mobile_pipeline_pcm16_push() {
        let config = MobileConfig::default();
        let pipeline = MobilePipeline::new(config);
        pipeline.push_pcm16(&[0, 16384, -16384]);
        assert_eq!(pipeline.audio_source().queued_samples(), 3);
    }
}
