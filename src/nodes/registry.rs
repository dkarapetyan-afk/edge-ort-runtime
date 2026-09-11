//! Capability → node registry built from a profile's manifests.

use super::{AsrNode, MtNode, SlmNode, TtsNode, VadNode};
use crate::profile::{CapabilityKind, Manifest};
use std::collections::HashMap;
use tracing::info;

#[derive(Default)]
pub struct NodeRegistry {
    pub vad: Option<VadNode>,
    pub asr: Option<AsrNode>,
    pub mt: Option<MtNode>,
    pub tts: Option<TtsNode>,
    pub slm: Option<SlmNode>,
}

impl NodeRegistry {
    pub fn from_manifests(manifests: HashMap<CapabilityKind, Manifest>) -> Self {
        let mut reg = Self {
            vad: None,
            asr: None,
            mt: None,
            tts: None,
            slm: None,
        };
        for (kind, man) in manifests {
            info!(
                capability = kind.as_str(),
                plug = %man.id,
                "registering capability node"
            );
            match kind {
                CapabilityKind::Vad => reg.vad = Some(VadNode::from_manifest(man)),
                CapabilityKind::Asr => reg.asr = Some(AsrNode::from_manifest(man)),
                CapabilityKind::Mt => reg.mt = Some(MtNode::from_manifest(man)),
                CapabilityKind::Tts => reg.tts = Some(TtsNode::from_manifest(man)),
                CapabilityKind::Slm => reg.slm = Some(SlmNode::from_manifest(man)),
            }
        }
        reg
    }

    pub fn has(&self, kind: CapabilityKind) -> bool {
        match kind {
            CapabilityKind::Vad => self.vad.is_some(),
            CapabilityKind::Asr => self.asr.is_some(),
            CapabilityKind::Mt => self.mt.is_some(),
            CapabilityKind::Tts => self.tts.is_some(),
            CapabilityKind::Slm => self.slm.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::EpPreference;
    use std::path::PathBuf;

    fn man(id: &str, cap: CapabilityKind) -> Manifest {
        Manifest {
            id: id.into(),
            capability: cap,
            model_path: PathBuf::from("missing.onnx"),
            inputs: vec!["in".into()],
            outputs: vec!["out".into()],
            sample_rate: 16_000,
            languages: vec![],
            ep_prefs: EpPreference::default_chain(),
            meta: Default::default(),
        }
    }

    #[test]
    fn registry_maps_capabilities() {
        let mut map = HashMap::new();
        map.insert(CapabilityKind::Asr, man("whisper-tiny", CapabilityKind::Asr));
        map.insert(CapabilityKind::Vad, man("silero-vad", CapabilityKind::Vad));
        let reg = NodeRegistry::from_manifests(map);
        assert!(reg.has(CapabilityKind::Asr));
        assert!(reg.has(CapabilityKind::Vad));
        assert!(!reg.has(CapabilityKind::Tts));
        assert_eq!(reg.asr.as_ref().unwrap().manifest.id, "whisper-tiny");
    }
}
