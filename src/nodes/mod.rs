//! Pluggable capability nodes: vad | asr | mt | tts.

mod asr;
mod mt;
mod registry;
mod traits;
mod tts;
mod vad;

pub use asr::AsrNode;
pub use mt::MtNode;
pub use registry::NodeRegistry;
pub use traits::{AsrOut, MtOut, NodeError, TtsOut, VadOut};
pub use tts::{playback_stub, TtsNode};
pub use vad::VadNode;
