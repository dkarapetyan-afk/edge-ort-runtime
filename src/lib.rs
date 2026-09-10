//! Pluggable PipeWire audio + ONNX Runtime inference runtime.
//!
//! Core types are capability-oriented (`vad | asr | mt | tts`). Concrete models
//! (Silero, Whisper, NLLB, Piper, …) are plugs described by JSON manifests —
//! they are never hard-coded into core enums.

pub mod audio;
pub mod nodes;
pub mod pipeline;
pub mod profile;
pub mod runtime;

pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent};
pub use profile::{CapabilityKind, Manifest, Profile};
pub use runtime::{probe_providers, EpPreference, ProviderInfo, SessionManager};
