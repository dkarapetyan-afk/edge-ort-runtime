//! Profile + model manifest loading.
//!
//! A **profile** maps capability kinds → manifest paths (plugs).
//! A **manifest** describes one ONNX model: paths, tensor names, languages, EP prefs.
//! Brand names (Whisper, NLLB, …) appear only as plug identifiers in JSON — never in core enums.

use crate::runtime::EpPreference;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub mod languages;
pub use languages::SupportedLanguage;

/// Capability kind — the only model-family taxonomy in core types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CapabilityKind {
    Vad,
    Asr,
    Mt,
    Tts,
    Slm,
}

impl CapabilityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vad => "vad",
            Self::Asr => "asr",
            Self::Mt => "mt",
            Self::Tts => "tts",
            Self::Slm => "slm",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "vad" => Some(Self::Vad),
            "asr" => Some(Self::Asr),
            "mt" => Some(Self::Mt),
            "tts" => Some(Self::Tts),
            "slm" => Some(Self::Slm),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

/// Per-model JSON manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Human / plug id (e.g. "silero-vad", "whisper-tiny", "tiny-asr-energy").
    pub id: String,
    pub capability: CapabilityKind,
    /// Path to .onnx (absolute or relative to the manifest file / profile root).
    pub model_path: PathBuf,
    /// Input tensor name(s). First is primary audio/text input.
    pub inputs: Vec<String>,
    /// Output tensor name(s).
    pub outputs: Vec<String>,
    /// Expected audio sample rate (for audio-in capabilities). Default 16000.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Languages this plug supports (BCP-47 / short codes). Empty = unspecified.
    #[serde(default)]
    pub languages: Vec<String>,
    /// EP preference order for this model.
    #[serde(default = "EpPreference::default_chain")]
    pub ep_prefs: Vec<EpPreference>,
    /// Extra free-form metadata (tokenizer path, vocab, etc.).
    #[serde(default)]
    pub meta: HashMap<String, serde_json::Value>,
}

fn default_sample_rate() -> u32 {
    16_000
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            id: String::new(),
            capability: CapabilityKind::Vad,
            model_path: PathBuf::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            sample_rate: default_sample_rate(),
            languages: Vec::new(),
            ep_prefs: EpPreference::default_chain(),
            meta: HashMap::new(),
        }
    }
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let text = fs::read_to_string(path)?;
        let mut m: Manifest = serde_json::from_str(&text)?;
        resolve_model_path(path, &mut m);
        Ok(m)
    }
}

fn resolve_model_path(manifest_path: &Path, m: &mut Manifest) {
    let Some(man_dir) = manifest_path.parent() else {
        return;
    };

    // 1) Relative to manifest directory.
    if m.model_path.is_relative() {
        let candidate = man_dir.join(&m.model_path);
        if candidate.exists() {
            m.model_path = candidate;
            return;
        }
    } else if m.model_path.exists() {
        return;
    }

    // 2) Relative to repo root guessed as ../../ from profiles/<name>/
    let fname = m
        .model_path
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| m.model_path.clone());
    let via_models = man_dir.join("../../models").join(&fname);
    if via_models.exists() {
        m.model_path = via_models;
        return;
    }

    // 3) Join original relative against repo root.
    let via_root = man_dir.join("../..").join(&m.model_path);
    if via_root.exists() {
        m.model_path = via_root;
        return;
    }

    // Leave as man_dir-relative even if missing (caller may stub).
    if m.model_path.is_relative() {
        m.model_path = man_dir.join(&m.model_path);
    }
}

/// Profile JSON: capability → manifest relative path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Map of capability kind string → manifest path (relative to profile file).
    pub nodes: HashMap<String, PathBuf>,
}

impl Profile {
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let text = fs::read_to_string(path)?;
        let p: Profile = serde_json::from_str(&text)?;
        Ok(p)
    }

    /// Resolve all manifests relative to the profile file location.
    pub fn load_manifests(
        &self,
        profile_path: &Path,
    ) -> Result<HashMap<CapabilityKind, Manifest>, ProfileError> {
        let base = profile_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let mut out = HashMap::new();
        for (cap_s, rel) in &self.nodes {
            let kind = CapabilityKind::parse(cap_s).ok_or_else(|| {
                ProfileError::Other(format!("unknown capability in profile: {cap_s}"))
            })?;
            let man_path = if rel.is_absolute() {
                rel.clone()
            } else {
                base.join(rel)
            };
            let manifest = Manifest::load(&man_path)?;
            if manifest.capability != kind {
                return Err(ProfileError::Other(format!(
                    "manifest {} capability {:?} does not match profile key {cap_s}",
                    man_path.display(),
                    manifest.capability
                )));
            }
            out.insert(kind, manifest);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_manifest_json() {
        let raw = r#"{
            "id": "tiny-asr-energy",
            "capability": "asr",
            "model_path": "models/tiny_asr_energy.onnx",
            "inputs": ["audio"],
            "outputs": ["logits"],
            "sample_rate": 16000,
            "languages": ["und"],
            "ep_prefs": ["cuda", "cpu"]
        }"#;
        let m: Manifest = serde_json::from_str(raw).unwrap();
        assert_eq!(m.id, "tiny-asr-energy");
        assert_eq!(m.capability, CapabilityKind::Asr);
        assert_eq!(m.ep_prefs, vec![EpPreference::Cuda, EpPreference::Cpu]);
        assert_eq!(m.inputs[0], "audio");
    }

    #[test]
    fn parse_profile_and_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let man_path = dir.path().join("asr.json");
        let mut f = fs::File::create(&man_path).unwrap();
        write!(
            f,
            r#"{{
                "id": "stub",
                "capability": "asr",
                "model_path": "missing.onnx",
                "inputs": ["audio"],
                "outputs": ["logits"]
            }}"#
        )
        .unwrap();

        let profile_path = dir.path().join("profile.json");
        let mut pf = fs::File::create(&profile_path).unwrap();
        write!(
            pf,
            r#"{{
                "name": "test",
                "nodes": {{ "asr": "asr.json" }}
            }}"#
        )
        .unwrap();

        let profile = Profile::load(&profile_path).unwrap();
        let maps = profile.load_manifests(&profile_path).unwrap();
        assert!(maps.contains_key(&CapabilityKind::Asr));
        assert_eq!(maps[&CapabilityKind::Asr].id, "stub");
    }

    #[test]
    fn test_parse_mobile_profile() {
        let mobile_profile_path = Path::new("profiles/mobile/profile.json");
        if mobile_profile_path.exists() {
            let profile = Profile::load(mobile_profile_path).unwrap();
            assert_eq!(profile.name, "mobile");
            let manifests = profile.load_manifests(mobile_profile_path).unwrap();
            assert!(manifests.contains_key(&CapabilityKind::Vad));
            assert!(manifests.contains_key(&CapabilityKind::Asr));
            assert!(manifests.contains_key(&CapabilityKind::Mt));

            let vad = &manifests[&CapabilityKind::Vad];
            assert!(vad.ep_prefs.contains(&EpPreference::Qnn));
            assert!(vad.ep_prefs.contains(&EpPreference::Nnapi));
            assert!(vad.ep_prefs.contains(&EpPreference::Xnnpack));

            let asr = &manifests[&CapabilityKind::Asr];
            assert!(asr.languages.contains(&"zh".to_string()));
            assert!(asr.languages.contains(&"ar".to_string()));
        }
    }
}
