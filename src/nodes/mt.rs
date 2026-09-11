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
            text: text.to_string(),
            src_lang: self.src_lang.clone(),
            tgt_lang: self.tgt_lang.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::CapabilityKind;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn mt_manifest(languages: Vec<String>) -> Manifest {
        Manifest {
            id: "test-mt".into(),
            capability: CapabilityKind::Mt,
            model_path: PathBuf::from("models/test_mt.onnx"),
            inputs: vec!["input_ids".into()],
            outputs: vec!["logits".into()],
            sample_rate: 16_000,
            languages,
            ep_prefs: vec![],
            meta: HashMap::new(),
        }
    }

    #[test]
    fn test_mt_manifest_language_pairs() {
        let node = MtNode::from_manifest(mt_manifest(vec!["fr".into(), "en".into()]));
        assert_eq!(node.src_lang, "fr");
        assert_eq!(node.tgt_lang, "en");
    }

    #[test]
    fn test_mt_manifest_fallback() {
        let empty_node = MtNode::from_manifest(mt_manifest(vec![]));
        assert_eq!(empty_node.src_lang, "auto");
        assert_eq!(empty_node.tgt_lang, "en");

        let single_node = MtNode::from_manifest(mt_manifest(vec!["es".into()]));
        assert_eq!(single_node.src_lang, "es");
        assert_eq!(single_node.tgt_lang, "en");
    }

    #[test]
    fn test_mt_process_non_empty_text() {
        let mut node = MtNode::from_manifest(mt_manifest(vec!["es".into(), "en".into()]));
        let out = node.process("Hola mundo").expect("process mt");
        assert_eq!(out.text, "Hola mundo");
        assert_eq!(out.src_lang, "es");
        assert_eq!(out.tgt_lang, "en");
    }

    #[test]
    fn test_mt_process_empty_text() {
        let mut node = MtNode::from_manifest(mt_manifest(vec!["fr".into(), "en".into()]));
        let out = node.process("").expect("process mt empty");
        assert_eq!(out.text, "");
        assert_eq!(out.src_lang, "fr");
        assert_eq!(out.tgt_lang, "en");
    }

    #[test]
    fn test_mt_process_multilingual_utf8() {
        let mut node = MtNode::from_manifest(mt_manifest(vec!["auto".into(), "en".into()]));
        let samples = [
            "¿Cómo estás?",
            "Merci beaucoup",
            "شكرا جزيلا",
            "नमस्ते दुनिया",
            "早上好",
            "こんにちは世界",
            "Շնորհակալություն",
            "Привет мир",
        ];
        for sample in samples {
            let out = node.process(sample).expect("process sample");
            assert_eq!(out.text, sample);
        }
    }
}
