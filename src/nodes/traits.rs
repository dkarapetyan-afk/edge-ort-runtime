//! Capability node traits (brand-agnostic).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("session: {0}")]
    Session(String),
    #[error("not ready: {0}")]
    NotReady(String),
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct VadOut {
    /// Probability / score that speech is present in the chunk.
    pub speech_prob: f32,
    pub is_speech: bool,
}

#[derive(Debug, Clone)]
pub struct AsrOut {
    pub text: String,
    /// English translation when dual-decode (Whisper translate) ran; None for English / unused.
    pub translation: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct MtOut {
    pub text: String,
    pub src_lang: String,
    pub tgt_lang: String,
}

#[derive(Debug, Clone)]
pub struct TtsOut {
    /// PCM mono f32 (playback stub may ignore sample rate).
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}
