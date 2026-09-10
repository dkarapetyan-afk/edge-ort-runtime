//! Canonical PCM bus: f32 mono @ 16 kHz.

pub const PCM_SAMPLE_RATE: u32 = 16_000;
pub const PCM_CHANNELS: u16 = 1;

/// A contiguous mono PCM buffer at [`PCM_SAMPLE_RATE`].
#[derive(Debug, Clone, Default)]
pub struct PcmBus {
    pub samples: Vec<f32>,
}

impl PcmBus {
    pub fn new(samples: Vec<f32>) -> Self {
        Self { samples }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / PCM_SAMPLE_RATE as f64
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
