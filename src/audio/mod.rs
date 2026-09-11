//! Audio capture and PCM bus utilities.
//!
//! The PCM bus is always **f32 mono @ 16 kHz**. Sources may capture at other
//! rates/channel counts; they are resampled and downmixed here.

mod fake;
mod pcm;
mod pipewire_playback;
#[cfg(feature = "pipewire")]
mod pipewire_source;
mod queue;
mod resample;
mod wav;

pub use fake::FakeSource;
pub use pcm::{PcmBus, PCM_CHANNELS, PCM_SAMPLE_RATE};
pub use pipewire_playback::{play_pcm, playback_log};
#[cfg(feature = "pipewire")]
pub use pipewire_source::{PipeWireSource, SourceKind};
pub use queue::{PushAudioSource, DEFAULT_CHUNK_SAMPLES};
pub use resample::{downmix_to_mono, resample_linear};
pub use wav::WavFileSource;

use thiserror::Error;

/// Trait for anything that produces PCM bus frames.
pub trait AudioSource: Send {
    fn name(&self) -> &str;
    /// Pull the next chunk of mono f32 @ 16 kHz samples. Returns empty on end-of-stream.
    fn pull(&mut self) -> Result<Vec<f32>, AudioError>;
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("pipewire: {0}")]
    PipeWire(String),
    #[error("source closed")]
    Closed,
    #[error("{0}")]
    Other(String),
}
