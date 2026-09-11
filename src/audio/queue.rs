//! Push-based AudioSource backed by a thread-safe queue.
//!
//! Ideal for external callbacks (Oboe / AAudio, JNI, WebSockets, streaming buffers)
//! where audio arrives asynchronously and is fed into the synchronous `Pipeline`.

use super::{
    downmix_to_mono, resample_linear, AudioError, AudioSource, PCM_SAMPLE_RATE,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const DEFAULT_CHUNK_SAMPLES: usize = 1600; // 100 ms @ 16 kHz

struct PushAudioSourceInner {
    queue: Mutex<VecDeque<f32>>,
    condvar: Condvar,
    closed: AtomicBool,
    name: String,
    chunk_samples: usize,
}

#[derive(Clone)]
pub struct PushAudioSource {
    inner: Arc<PushAudioSourceInner>,
}

impl PushAudioSource {
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_chunk_samples(name, DEFAULT_CHUNK_SAMPLES)
    }

    pub fn with_chunk_samples(name: impl Into<String>, chunk_samples: usize) -> Self {
        Self {
            inner: Arc::new(PushAudioSourceInner {
                queue: Mutex::new(VecDeque::new()),
                condvar: Condvar::new(),
                closed: AtomicBool::new(false),
                name: name.into(),
                chunk_samples: chunk_samples.max(1),
            }),
        }
    }

    /// Push mono f32 @ 16 kHz samples directly into the buffer.
    pub fn push(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let mut q = self.inner.queue.lock().unwrap();
        q.extend(samples);
        self.inner.condvar.notify_one();
    }

    /// Convert 16-bit signed PCM to f32 normalized (-1.0 to 1.0) and push.
    pub fn push_pcm16(&self, pcm16: &[i16]) {
        let f32_samples: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
        self.push(&f32_samples);
    }

    /// Downmix multi-channel audio to mono, resample to 16 kHz, and push.
    pub fn push_resampled(&self, samples: &[f32], input_rate: u32, channels: u16) {
        if samples.is_empty() {
            return;
        }
        let mono = downmix_to_mono(samples, channels as usize);
        let resampled = resample_linear(&mono, input_rate, PCM_SAMPLE_RATE);
        self.push(&resampled);
    }

    /// Convert 16-bit PCM multi-channel audio from arbitrary sample rate to 16 kHz mono and push.
    pub fn push_pcm16_resampled(&self, pcm16: &[i16], input_rate: u32, channels: u16) {
        if pcm16.is_empty() {
            return;
        }
        let f32_samples: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
        self.push_resampled(&f32_samples, input_rate, channels);
    }

    /// Signal end of audio stream.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.condvar.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }

    pub fn queued_samples(&self) -> usize {
        self.inner.queue.lock().unwrap().len()
    }
}

impl AudioSource for PushAudioSource {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn pull(&mut self) -> Result<Vec<f32>, AudioError> {
        let mut q = self.inner.queue.lock().unwrap();

        loop {
            if q.len() >= self.inner.chunk_samples {
                let n = self.inner.chunk_samples;
                return Ok(q.drain(..n).collect());
            }

            if self.inner.closed.load(Ordering::SeqCst) {
                if q.is_empty() {
                    return Ok(Vec::new());
                }
                let count = q.len().min(self.inner.chunk_samples);
                return Ok(q.drain(..count).collect());
            }

            let (new_q, timeout_res) = self
                .inner
                .condvar
                .wait_timeout(q, Duration::from_millis(50))
                .unwrap();
            q = new_q;

            if timeout_res.timed_out() {
                if q.is_empty() {
                    if self.inner.closed.load(Ordering::SeqCst) {
                        return Ok(Vec::new());
                    }
                    return Ok(vec![0.0; self.inner.chunk_samples]);
                } else {
                    let count = q.len().min(self.inner.chunk_samples);
                    return Ok(q.drain(..count).collect());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_pull_exact_chunk() {
        let mut source = PushAudioSource::with_chunk_samples("test", 100);
        assert_eq!(source.name(), "test");
        assert_eq!(source.queued_samples(), 0);

        let data: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        source.push(&data);
        assert_eq!(source.queued_samples(), 100);

        let pulled = source.pull().unwrap();
        assert_eq!(pulled.len(), 100);
        assert_eq!(pulled[0], 0.0);
        assert_eq!(source.queued_samples(), 0);
    }

    #[test]
    fn test_push_pcm16_normalization() {
        let mut source = PushAudioSource::with_chunk_samples("pcm16", 4);
        let pcm: [i16; 4] = [0, 16384, -16384, 32767];
        source.push_pcm16(&pcm);

        let pulled = source.pull().unwrap();
        assert_eq!(pulled.len(), 4);
        assert!((pulled[0] - 0.0).abs() < 1e-4);
        assert!((pulled[1] - 0.5).abs() < 1e-4);
        assert!((pulled[2] - (-0.5)).abs() < 1e-4);
        assert!((pulled[3] - (32767.0 / 32768.0)).abs() < 1e-4);
    }

    #[test]
    fn test_push_resampled_stereo_48k() {
        let mut source = PushAudioSource::with_chunk_samples("resampled", 16);
        let mut stereo_48k = Vec::new();
        for _ in 0..48 {
            stereo_48k.push(0.5f32);
            stereo_48k.push(0.5f32);
        }
        source.push_resampled(&stereo_48k, 48000, 2);

        let pulled = source.pull().unwrap();
        assert_eq!(pulled.len(), 16);
        for s in pulled {
            assert!((s - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_close_and_eos() {
        let mut source = PushAudioSource::with_chunk_samples("close_test", 50);
        source.push(&[0.1; 20]);
        assert!(!source.is_closed());

        source.close();
        assert!(source.is_closed());

        let remaining = source.pull().unwrap();
        assert_eq!(remaining.len(), 20);

        let eos = source.pull().unwrap();
        assert!(eos.is_empty());
    }
}
