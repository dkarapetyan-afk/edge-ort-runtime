//! Pluggable PipeWire audio + ONNX Runtime inference runtime.
//!
//! Core types are capability-oriented (`vad | asr | mt | tts`). Concrete models
//! (Silero, Whisper, NLLB, Piper, …) are plugs described by JSON manifests —
//! they are never hard-coded into core enums.
//!
//! # Example
//! ```rust
//! use edge_ort_runtime::profile::SupportedLanguage;
//!
//! let lang = SupportedLanguage::from_code("fr").unwrap();
//! assert_eq!(lang.code(), "fr");
//! assert_eq!(lang.label(), "French");
//! ```

#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod audio;
pub mod mobile;
pub mod nodes;
pub mod pipeline;
pub mod profile;
pub mod runtime;

pub use mobile::{MobileConfig, MobilePipeline};
pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent};
pub use profile::{CapabilityKind, Manifest, Profile, SupportedLanguage};
pub use runtime::{probe_providers, EpPreference, ProviderInfo, SessionManager};
