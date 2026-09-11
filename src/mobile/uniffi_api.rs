//! UniFFI exports and interfaces for Android Kotlin and iOS Swift bindings.

use crate::mobile::{MobileConfig, MobilePipeline};
use crate::pipeline::PipelineEvent;
use crate::profile::SupportedLanguage;
use crate::runtime::probe_providers;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Speech event listener callback interface implemented in Kotlin.
#[cfg_attr(feature = "uniffi", uniffi::export(callback_interface))]
pub trait NativeSpeechListener: Send + Sync {
    fn on_status(&self, status: String);
    fn on_vad(&self, speech_prob: f32, is_speech: bool);
    fn on_transcript(&self, text: String, translation: Option<String>, confidence: f32);
    fn on_error(&self, error: String);
    fn on_done(&self);
}

/// Metadata record for a supported language.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct LanguageInfo {
    pub code: String,
    pub label: String,
    pub is_rtl: bool,
    pub is_cjk: bool,
}

/// Metadata record for an available execution provider.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ProviderInfoRecord {
    pub name: String,
    pub key: String,
    pub available: bool,
    pub platform_ok: bool,
}

#[derive(Debug, thiserror::Error)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Error))]
pub enum NativePipelineError {
    #[error("{msg}")]
    ExecutionError { msg: String },
}

/// Native mobile pipeline handle for Android / iOS.
#[cfg_attr(feature = "uniffi", derive(uniffi::Object))]
pub struct NativePipelineHandle {
    inner: Mutex<MobilePipeline>,
}

#[cfg_attr(feature = "uniffi", uniffi::export)]
impl NativePipelineHandle {
    #[cfg_attr(feature = "uniffi", uniffi::constructor)]
    pub fn new(
        language_code: String,
        enable_vad: bool,
        enable_mt: bool,
        profile_path: Option<String>,
    ) -> Result<Arc<Self>, NativePipelineError> {
        let language = SupportedLanguage::from_code(&language_code)
            .unwrap_or(SupportedLanguage::Auto);

        let config = MobileConfig {
            language,
            enable_vad,
            enable_mt,
            profile_path: profile_path.map(PathBuf::from),
        };

        Ok(Arc::new(Self {
            inner: Mutex::new(MobilePipeline::new(config)),
        }))
    }

    /// Push mono f32 @ 16 kHz samples.
    pub fn push_samples(&self, samples: Vec<f32>) {
        if let Ok(guard) = self.inner.lock() {
            guard.push_samples(&samples);
        }
    }

    /// Push 16-bit signed PCM samples.
    pub fn push_pcm16(&self, samples: Vec<i16>) {
        if let Ok(guard) = self.inner.lock() {
            guard.push_pcm16(&samples);
        }
    }

    /// Push multi-channel f32 samples from microphone (e.g. 48 kHz stereo); automatically downmixes and resamples to 16 kHz mono.
    pub fn push_resampled(&self, samples: Vec<f32>, sample_rate: u32, channels: u16) {
        if let Ok(guard) = self.inner.lock() {
            guard.push_resampled(&samples, sample_rate, channels);
        }
    }

    /// Push multi-channel 16-bit PCM samples from microphone (e.g. 48 kHz stereo); automatically downmixes and resamples to 16 kHz mono.
    pub fn push_pcm16_resampled(&self, samples: Vec<i16>, sample_rate: u32, channels: u16) {
        if let Ok(guard) = self.inner.lock() {
            guard.push_pcm16_resampled(&samples, sample_rate, channels);
        }
    }

    /// Start processing audio. Delivers events to `listener`.
    pub fn start(&self, listener: Box<dyn NativeSpeechListener>) -> Result<(), NativePipelineError> {
        let mut guard = self.inner.lock().map_err(|e| NativePipelineError::ExecutionError {
            msg: e.to_string(),
        })?;

        guard
            .start(move |ev| {
                match ev {
                    PipelineEvent::Status(s) => listener.on_status(s),
                    PipelineEvent::Vad { speech_prob, is_speech } => {
                        listener.on_vad(speech_prob, is_speech);
                    }
                    PipelineEvent::Transcript { text, translation, confidence } => {
                        listener.on_transcript(text, translation, confidence);
                    }
                    PipelineEvent::Error(err) => listener.on_error(err),
                    PipelineEvent::Done => listener.on_done(),
                    PipelineEvent::EpInfo { .. } => {}
                }
                true
            })
            .map_err(|e| NativePipelineError::ExecutionError { msg: e.to_string() })
    }

    /// Stop audio pipeline.
    pub fn stop(&self) -> Result<(), NativePipelineError> {
        let mut guard = self.inner.lock().map_err(|e| NativePipelineError::ExecutionError {
            msg: e.to_string(),
        })?;
        guard.stop().map_err(|e| NativePipelineError::ExecutionError { msg: e.to_string() })
    }

    /// Returns true if background worker is currently active.
    pub fn is_running(&self) -> bool {
        self.inner.lock().map(|g| g.is_running()).unwrap_or(false)
    }

    /// Clean filler words and speech disfluencies.
    pub fn clean_text(&self, text: String) -> String {
        crate::nodes::clean_transcript_heuristics(&text)
    }

    /// Summarize speech transcript into action items and bullet points.
    pub fn summarize_text(&self, text: String) -> NativeSlmSummary {
        summarize_transcript_text(text)
    }
}

/// Metadata record for an SLM summary output.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct NativeSlmSummary {
    pub cleaned_text: String,
    pub bullet_points: Vec<String>,
    pub word_count_original: u32,
    pub word_count_cleaned: u32,
}

/// Clean speech transcript text directly without an active pipeline instance.
#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn clean_transcript_text(text: String) -> String {
    crate::nodes::clean_transcript_heuristics(&text)
}

/// Summarize transcript text into bullet points and action items.
#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn summarize_transcript_text(text: String) -> NativeSlmSummary {
    let bullets = crate::nodes::extract_bullet_points(&text);
    let cleaned = crate::nodes::clean_transcript_heuristics(&text);
    let orig_words = text.split_whitespace().count() as u32;
    let clean_words = cleaned.split_whitespace().count() as u32;
    NativeSlmSummary {
        cleaned_text: cleaned,
        bullet_points: bullets,
        word_count_original: orig_words,
        word_count_cleaned: clean_words,
    }
}

/// Returns list of all 16 supported languages with metadata for Android UI spinners/pickers.
#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn get_supported_languages() -> Vec<LanguageInfo> {
    SupportedLanguage::ALL
        .iter()
        .map(|l| LanguageInfo {
            code: l.code().to_string(),
            label: l.label().to_string(),
            is_rtl: l.is_rtl(),
            is_cjk: l.is_cjk(),
        })
        .collect()
}

/// Returns list of neural execution providers and their availability status on this device.
#[cfg_attr(feature = "uniffi", uniffi::export)]
pub fn get_execution_providers() -> Vec<ProviderInfoRecord> {
    probe_providers()
        .into_iter()
        .map(|p| ProviderInfoRecord {
            name: p.name,
            key: p.preference.as_str().to_string(),
            available: p.available,
            platform_ok: p.platform_ok,
        })
        .collect()
}

/// Direct C-ABI entry point for Google Oboe / AAudio C++ callbacks.
/// Avoids JNI overhead when streaming high-frequency audio buffers.
///
/// # Safety
/// Caller must ensure `handle` and `samples` are non-null and point to valid memory.
#[no_mangle]
pub unsafe extern "C" fn edge_ort_push_pcm16(
    handle: *const NativePipelineHandle,
    samples: *const i16,
    num_samples: usize,
    sample_rate: u32,
    channels: u16,
) -> i32 {
    if handle.is_null() || samples.is_null() || num_samples == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(samples, num_samples);
    (*handle).push_pcm16_resampled(slice.to_vec(), sample_rate, channels);
    0
}

/// Direct C-ABI entry point for float audio buffers.
///
/// # Safety
/// Caller must ensure `handle` and `samples` are non-null and point to valid memory.
#[no_mangle]
pub unsafe extern "C" fn edge_ort_push_f32(
    handle: *const NativePipelineHandle,
    samples: *const f32,
    num_samples: usize,
    sample_rate: u32,
    channels: u16,
) -> i32 {
    if handle.is_null() || samples.is_null() || num_samples == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(samples, num_samples);
    (*handle).push_resampled(slice.to_vec(), sample_rate, channels);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct TestListener {
        received_status: Arc<AtomicBool>,
    }

    impl NativeSpeechListener for TestListener {
        fn on_status(&self, status: String) {
            if status.contains("pipeline start") {
                self.received_status.store(true, Ordering::SeqCst);
            }
        }
        fn on_vad(&self, _speech_prob: f32, _is_speech: bool) {}
        fn on_transcript(&self, _text: String, _translation: Option<String>, _confidence: f32) {}
        fn on_error(&self, _error: String) {}
        fn on_done(&self) {}
    }

    #[test]
    fn test_native_pipeline_handle_lifecycle() {
        let handle = NativePipelineHandle::new("es".into(), true, true, None).unwrap();
        assert!(!handle.is_running());

        let flag = Arc::new(AtomicBool::new(false));
        let listener = Box::new(TestListener {
            received_status: Arc::clone(&flag),
        });

        handle.start(listener).unwrap();
        assert!(handle.is_running());

        // Wait for pipeline start status event
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while !flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Push some silence
        handle.push_samples(vec![0.0; 1600]);
        handle.push_pcm16(vec![0; 1600]);

        handle.stop().unwrap();
        assert!(!handle.is_running());
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_get_supported_languages_export() {
        let langs = get_supported_languages();
        assert_eq!(langs.len(), 16);
        let ar = langs.iter().find(|l| l.code == "ar").unwrap();
        assert!(ar.is_rtl);
        let zh = langs.iter().find(|l| l.code == "zh").unwrap();
        assert!(zh.is_cjk);
    }

    #[test]
    fn test_get_execution_providers_export() {
        let providers = get_execution_providers();
        assert!(!providers.is_empty());
        assert!(providers.iter().any(|p| p.key == "cpu" && p.available));
    }

    #[test]
    fn test_c_abi_push_pcm16_and_f32() {
        let handle = NativePipelineHandle::new("auto".into(), true, true, None).unwrap();
        let handle_ptr = Arc::as_ptr(&handle);

        let pcm = [0i16, 16384, -16384, 32767];
        unsafe {
            let res = edge_ort_push_pcm16(handle_ptr, pcm.as_ptr(), pcm.len(), 16000, 1);
            assert_eq!(res, 0);

            let null_res = edge_ort_push_pcm16(std::ptr::null(), pcm.as_ptr(), pcm.len(), 16000, 1);
            assert_eq!(null_res, -1);
        }

        let f32_samples = [0.0f32, 0.5, -0.5, 1.0];
        unsafe {
            let res = edge_ort_push_f32(handle_ptr, f32_samples.as_ptr(), f32_samples.len(), 16000, 1);
            assert_eq!(res, 0);

            let empty_res = edge_ort_push_f32(handle_ptr, f32_samples.as_ptr(), 0, 16000, 1);
            assert_eq!(empty_res, -1);
        }

        assert_eq!(handle.inner.lock().unwrap().audio_source().queued_samples(), 8);
    }
}
