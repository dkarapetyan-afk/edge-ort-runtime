//! Synthetic / file-free audio source for demos without PipeWire hardware.

use super::{AudioError, AudioSource, PCM_SAMPLE_RATE};
use std::f32::consts::PI;

/// Generates a quiet sine tone (or silence) at the PCM bus rate.
pub struct FakeSource {
    phase: f32,
    freq_hz: f32,
    chunk_samples: usize,
    remaining_chunks: Option<usize>,
    silent: bool,
}

impl FakeSource {
    pub fn sine(secs: f32) -> Self {
        let total = (secs * PCM_SAMPLE_RATE as f32) as usize;
        let chunk = PCM_SAMPLE_RATE as usize / 10; // 100 ms
        Self {
            phase: 0.0,
            freq_hz: 440.0,
            chunk_samples: chunk,
            remaining_chunks: Some((total / chunk).max(1)),
            silent: false,
        }
    }

    /// Continuous sine for GUI / long-running demos (stop via pipeline stop flag).
    pub fn sine_forever() -> Self {
        Self {
            phase: 0.0,
            freq_hz: 440.0,
            chunk_samples: PCM_SAMPLE_RATE as usize / 10,
            remaining_chunks: None,
            silent: false,
        }
    }

    pub fn silence_forever() -> Self {
        Self {
            phase: 0.0,
            freq_hz: 0.0,
            chunk_samples: PCM_SAMPLE_RATE as usize / 10,
            remaining_chunks: None,
            silent: true,
        }
    }
}

impl AudioSource for FakeSource {
    fn name(&self) -> &str {
        if self.silent {
            "fake:silence"
        } else if self.remaining_chunks.is_none() {
            "fake:sine-forever"
        } else {
            "fake:sine"
        }
    }

    fn pull(&mut self) -> Result<Vec<f32>, AudioError> {
        if let Some(left) = self.remaining_chunks.as_mut() {
            if *left == 0 {
                return Ok(Vec::new());
            }
            *left -= 1;
        }
        let n = self.chunk_samples;
        if self.silent {
            // Brief sleep so forever-silence doesn't busy-spin the pipeline.
            std::thread::sleep(std::time::Duration::from_millis(80));
            return Ok(vec![0.0; n]);
        }
        let mut buf = Vec::with_capacity(n);
        let sr = PCM_SAMPLE_RATE as f32;
        for _ in 0..n {
            let s = (2.0 * PI * self.freq_hz * self.phase / sr).sin() * 0.1;
            buf.push(s);
            self.phase += 1.0;
            if self.phase >= sr {
                self.phase -= sr;
            }
        }
        // Pace forever sources roughly real-time.
        if self.remaining_chunks.is_none() {
            std::thread::sleep(std::time::Duration::from_millis(80));
        }
        Ok(buf)
    }
}
