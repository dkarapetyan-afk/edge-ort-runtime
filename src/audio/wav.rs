//! WAV file audio source (16-bit PCM preferred; resampled/downmixed to bus).

use super::{
    downmix_to_mono, resample_linear, AudioError, AudioSource, PCM_SAMPLE_RATE,
};
use std::path::{Path, PathBuf};

/// Reads a WAV once and yields mono f32 @ 16 kHz in one or more chunks.
pub struct WavFileSource {
    name: String,
    samples: Vec<f32>,
    pos: usize,
    /// Max samples per pull (default: entire file).
    chunk_samples: usize,
}

impl WavFileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let path = path.as_ref();
        let reader = hound::WavReader::open(path)
            .map_err(|e| AudioError::Other(format!("open {}: {e}", path.display())))?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;
        let sr = spec.sample_rate;

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = match spec.bits_per_sample {
                    8 => i8::MAX as f32,
                    16 => i16::MAX as f32,
                    24 => ((1i32 << 23) - 1) as f32,
                    32 => i32::MAX as f32,
                    b => {
                        return Err(AudioError::Other(format!(
                            "unsupported bits_per_sample={b}"
                        )));
                    }
                };
                reader
                    .into_samples::<i32>()
                    .map(|s| {
                        s.map(|v| v as f32 / max)
                            .map_err(|e| AudioError::Other(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .map(|s| s.map_err(|e| AudioError::Other(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let mono = if channels == 1 {
            interleaved
        } else {
            downmix_to_mono(&interleaved, channels)
        };

        let samples = if sr == PCM_SAMPLE_RATE {
            mono
        } else {
            resample_linear(&mono, sr, PCM_SAMPLE_RATE)
        };

        let n = samples.len().max(1);
        Ok(Self {
            name: format!("file:{}", path.display()),
            samples,
            pos: 0,
            chunk_samples: n,
        })
    }

    /// Prefer pulling the whole utterance (good for Whisper).
    pub fn with_chunk_samples(mut self, n: usize) -> Self {
        self.chunk_samples = n.max(1);
        self
    }

    pub fn path_name(path: &Path) -> PathBuf {
        path.to_path_buf()
    }

    pub fn len_samples(&self) -> usize {
        self.samples.len()
    }
}

impl AudioSource for WavFileSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn pull(&mut self) -> Result<Vec<f32>, AudioError> {
        if self.pos >= self.samples.len() {
            return Ok(Vec::new());
        }
        let end = (self.pos + self.chunk_samples).min(self.samples.len());
        let out = self.samples[self.pos..end].to_vec();
        self.pos = end;
        Ok(out)
    }
}
