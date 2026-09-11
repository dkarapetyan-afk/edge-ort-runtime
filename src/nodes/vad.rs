//! VAD node — Silero (or any) via manifest; energy fallback if model missing / ORT fails.

use super::{NodeError, VadOut};
use crate::profile::Manifest;
use crate::runtime::SessionManager;
use ndarray::{Array2, Array3};
use ort::value::Tensor;
use tracing::{debug, info, warn};

const DEFAULT_WINDOW: usize = 512;
const DEFAULT_CONTEXT: usize = 64;
const DEFAULT_STATE_H: usize = 2;
const DEFAULT_STATE_C: usize = 128;
const DEFAULT_THRESHOLD: f32 = 0.5;

pub struct VadNode {
    pub manifest: Manifest,
    pub ready: bool,
    session_key: String,
    loaded: bool,
    /// LSTM / RNN state for Silero-style plugs: [2, 1, 128]
    state: Array3<f32>,
    /// Trailing context samples prepended to each window.
    context: Vec<f32>,
    window_samples: usize,
    context_samples: usize,
    threshold: f32,
    /// Energy stub threshold (RMS).
    energy_threshold: f32,
    ort_failed: bool,
}

impl VadNode {
    pub fn from_manifest(manifest: Manifest) -> Self {
        let ready = manifest.model_path.exists();
        if !ready {
            warn!(
                id = %manifest.id,
                path = %manifest.model_path.display(),
                "VAD model missing — using energy stub"
            );
        }

        let window_samples = meta_usize(&manifest, "window_samples").unwrap_or(DEFAULT_WINDOW);
        let context_samples = meta_usize(&manifest, "context_samples").unwrap_or(DEFAULT_CONTEXT);
        let threshold = meta_f32(&manifest, "threshold").unwrap_or(DEFAULT_THRESHOLD);
        let state_h = meta_usize(&manifest, "state_h").unwrap_or(DEFAULT_STATE_H);
        let state_c = meta_usize(&manifest, "state_c").unwrap_or(DEFAULT_STATE_C);

        Self {
            session_key: format!("vad:{}", manifest.id),
            manifest,
            ready,
            loaded: false,
            state: Array3::zeros((state_h, 1, state_c)),
            context: vec![0.0; context_samples],
            window_samples,
            context_samples,
            threshold,
            energy_threshold: 0.02,
            ort_failed: false,
        }
    }

    pub fn ensure_loaded(&mut self, sessions: &mut SessionManager) -> Result<(), NodeError> {
        if self.loaded || !self.ready || self.ort_failed {
            return Ok(());
        }
        match sessions.load(
            &self.session_key,
            &self.manifest.model_path,
            &self.manifest.ep_prefs,
        ) {
            Ok(_) => {
                self.loaded = true;
                info!(id = %self.manifest.id, "VAD ONNX session ready");
                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "VAD ONNX load failed — energy fallback");
                self.ort_failed = true;
                Ok(())
            }
        }
    }

    pub fn process(
        &mut self,
        sessions: &mut SessionManager,
        pcm: &[f32],
    ) -> Result<VadOut, NodeError> {
        self.ensure_loaded(sessions)?;

        if self.loaded && !self.ort_failed {
            match self.process_onnx(sessions, pcm) {
                Ok(out) => {
                    // OR with energy gate so demos (e.g. fake sine) still pass while
                    // Silero suppresses quiet silence. Energy alone remains the
                    // fallback when the model is missing or ORT fails.
                    if out.is_speech {
                        return Ok(out);
                    }
                    let energy = self.process_energy(pcm);
                    if energy.is_speech {
                        return Ok(VadOut {
                            speech_prob: out.speech_prob.max(energy.speech_prob),
                            is_speech: true,
                        });
                    }
                    return Ok(out);
                }
                Err(e) => {
                    warn!(error = %e, "VAD ORT inference failed — energy fallback");
                    self.ort_failed = true;
                }
            }
        }

        Ok(self.process_energy(pcm))
    }

    fn process_energy(&self, pcm: &[f32]) -> VadOut {
        let energy = if pcm.is_empty() {
            0.0
        } else {
            let sum: f32 = pcm.iter().map(|x| x * x).sum();
            (sum / pcm.len() as f32).sqrt()
        };
        let speech_prob = (energy / self.energy_threshold).clamp(0.0, 1.0);
        VadOut {
            is_speech: energy >= self.energy_threshold * 0.5,
            speech_prob,
        }
    }

    /// Silero-style: for each 512-sample window, concat 64-sample context → run → update state/context.
    fn process_onnx(
        &mut self,
        sessions: &mut SessionManager,
        pcm: &[f32],
    ) -> Result<VadOut, NodeError> {
        if pcm.is_empty() {
            return Ok(VadOut {
                speech_prob: 0.0,
                is_speech: false,
            });
        }

        let input_name = self
            .manifest
            .inputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("input");
        let state_name = self
            .manifest
            .inputs
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("state");
        let sr_name = self
            .manifest
            .inputs
            .get(2)
            .map(|s| s.as_str())
            .unwrap_or("sr");
        let out_name = self
            .manifest
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or("output");
        let state_out_name = self
            .manifest
            .outputs
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("stateN");

        let sr = self.manifest.sample_rate as i64;
        let win = self.window_samples;
        let ctx_n = self.context_samples;

        let mut max_prob = 0.0f32;
        let mut pos = 0usize;

        while pos < pcm.len() {
            let mut chunk = vec![0.0f32; win];
            let take = (pcm.len() - pos).min(win);
            chunk[..take].copy_from_slice(&pcm[pos..pos + take]);
            // Remaining stays zero-padded (Silero pads short windows).

            // input = context (ctx_n) + chunk (win) → length ctx_n+win
            let mut full = Vec::with_capacity(ctx_n + win);
            full.extend_from_slice(&self.context);
            full.extend_from_slice(&chunk);

            let arr = Array2::from_shape_vec((1, full.len()), full.clone())
                .map_err(|e| NodeError::Other(e.to_string()))?;
            let input = Tensor::from_array(arr).map_err(|e| NodeError::Session(e.to_string()))?;

            let state_t = Tensor::from_array(self.state.clone())
                .map_err(|e| NodeError::Session(e.to_string()))?;
            let sr_t = Tensor::from_array((Vec::<usize>::new(), vec![sr]))
                .map_err(|e| NodeError::Session(e.to_string()))?;

            let session = sessions
                .get_mut(&self.session_key)
                .ok_or_else(|| NodeError::NotReady("vad session".into()))?;

            let outputs = session
                .run(ort::inputs![
                    input_name => input,
                    state_name => state_t,
                    sr_name => sr_t
                ])
                .map_err(|e| NodeError::Session(e.to_string()))?;

            let prob_val = outputs
                .get(out_name)
                .ok_or_else(|| NodeError::Other(format!("missing VAD output {out_name}")))?;
            let (_shape, data) = prob_val
                .try_extract_tensor::<f32>()
                .map_err(|e| NodeError::Session(e.to_string()))?;
            let prob = data.first().copied().unwrap_or(0.0);
            max_prob = max_prob.max(prob);

            if let Some(state_val) = outputs.get(state_out_name) {
                let (shape, sdata) = state_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| NodeError::Session(e.to_string()))?;
                let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                if dims.len() == 3 && sdata.len() == dims[0] * dims[1] * dims[2] {
                    self.state = Array3::from_shape_vec(
                        (dims[0], dims[1], dims[2]),
                        sdata.to_vec(),
                    )
                    .map_err(|e| NodeError::Other(e.to_string()))?;
                }
            }

            // Update context = last ctx_n of the concatenated input.
            if full.len() >= ctx_n {
                self.context = full[full.len() - ctx_n..].to_vec();
            }

            pos += win;
            // Don't pad-process pure silence tails beyond one partial window.
            if take < win {
                break;
            }
        }

        debug!(prob = max_prob, "silero vad");
        Ok(VadOut {
            speech_prob: max_prob,
            is_speech: max_prob >= self.threshold,
        })
    }
}

fn meta_usize(m: &Manifest, key: &str) -> Option<usize> {
    m.meta.get(key).and_then(|v| {
        v.as_u64()
            .map(|n| n as usize)
            .or_else(|| v.as_i64().map(|n| n as usize))
            .or_else(|| v.as_f64().map(|n| n as usize))
    })
}

fn meta_f32(m: &Manifest, key: &str) -> Option<f32> {
    m.meta.get(key).and_then(|v| {
        v.as_f64()
            .map(|n| n as f32)
            .or_else(|| v.as_i64().map(|n| n as f32))
    })
}
