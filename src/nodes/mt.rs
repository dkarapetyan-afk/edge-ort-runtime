//! Machine-translation node — stub pending NLLB (or other) ONNX plug.

use super::{MtOut, NodeError};
use crate::profile::Manifest;
use tracing::warn;

pub struct MtNode {
    pub manifest: Manifest,
    pub src_lang: String,
    pub tgt_lang: String,
}

impl MtNode {
    pub fn from_manifest(manifest: Manifest) -> Self {
        if !manifest.model_path.exists() {
            warn!(
                id = %manifest.id,
                path = %manifest.model_path.display(),
                "MT model missing — passthrough stub"
            );
        }
        let src = manifest
            .languages
            .first()
            .cloned()
            .unwrap_or_else(|| "auto".into());
        let tgt = manifest
            .languages
            .get(1)
            .cloned()
            .unwrap_or_else(|| "en".into());
        Self {
            manifest,
            src_lang: src,
            tgt_lang: tgt,
        }
    }

    /// TODO: run MT ONNX. For now, passthrough with a marker when non-empty.
    pub fn process(&mut self, text: &str) -> Result<MtOut, NodeError> {
        if text.is_empty() {
            return Ok(MtOut {
                text: String::new(),
                src_lang: self.src_lang.clone(),
                tgt_lang: self.tgt_lang.clone(),
            });
        }
        Ok(MtOut {
            text: format!("{text}"),
            src_lang: self.src_lang.clone(),
            tgt_lang: self.tgt_lang.clone(),
        })
    }
}
