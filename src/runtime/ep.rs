//! Execution Provider priority chain with graceful degradation.

use ort::ep::{CPU, ExecutionProvider, ExecutionProviderDispatch};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Named EP preference used in manifests / profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpPreference {
    TensorRt,
    Cuda,
    Rocm,
    OpenVino,
    Qnn,
    Nnapi,
    Xnnpack,
    Cpu,
}

impl EpPreference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TensorRt => "tensorrt",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::OpenVino => "openvino",
            Self::Qnn => "qnn",
            Self::Nnapi => "nnapi",
            Self::Xnnpack => "xnnpack",
            Self::Cpu => "cpu",
        }
    }

    /// Default priority: TensorRT → CUDA → ROCm → OpenVINO → QNN → NNAPI → XNNPACK → CPU.
    pub fn default_chain() -> Vec<Self> {
        vec![
            Self::TensorRt,
            Self::Cuda,
            Self::Rocm,
            Self::OpenVino,
            Self::Qnn,
            Self::Nnapi,
            Self::Xnnpack,
            Self::Cpu,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub preference: EpPreference,
    pub available: bool,
    pub platform_ok: bool,
}

fn dispatch_for(pref: EpPreference) -> Option<ExecutionProviderDispatch> {
    match pref {
        EpPreference::Cpu => Some(CPU::default().build()),
        #[cfg(feature = "ort-cuda")]
        EpPreference::Cuda => Some(ort::ep::CUDA::default().build()),
        #[cfg(feature = "ort-tensorrt")]
        EpPreference::TensorRt => Some(ort::ep::TensorRT::default().build()),
        #[cfg(feature = "ort-rocm")]
        EpPreference::Rocm => Some(ort::ep::ROCm::default().build()),
        #[cfg(feature = "ort-openvino")]
        EpPreference::OpenVino => Some(ort::ep::OpenVINO::default().build()),
        #[cfg(feature = "ort-qnn")]
        EpPreference::Qnn => Some(ort::ep::QNN::default().build()),
        #[cfg(feature = "ort-nnapi")]
        EpPreference::Nnapi => Some(ort::ep::NNAPI::default().build()),
        #[cfg(feature = "ort-xnnpack")]
        EpPreference::Xnnpack => Some(ort::ep::XNNPACK::default().build()),
        #[cfg(not(feature = "ort-cuda"))]
        EpPreference::Cuda => None,
        #[cfg(not(feature = "ort-tensorrt"))]
        EpPreference::TensorRt => None,
        #[cfg(not(feature = "ort-rocm"))]
        EpPreference::Rocm => None,
        #[cfg(not(feature = "ort-openvino"))]
        EpPreference::OpenVino => None,
        #[cfg(not(feature = "ort-qnn"))]
        EpPreference::Qnn => None,
        #[cfg(not(feature = "ort-nnapi"))]
        EpPreference::Nnapi => None,
        #[cfg(not(feature = "ort-xnnpack"))]
        EpPreference::Xnnpack => None,
    }
}

fn probe_one(pref: EpPreference) -> ProviderInfo {
    let (available, platform_ok, name) = match pref {
        EpPreference::Cpu => {
            let ep = CPU::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-cuda")]
        EpPreference::Cuda => {
            let ep = ort::ep::CUDA::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-tensorrt")]
        EpPreference::TensorRt => {
            let ep = ort::ep::TensorRT::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-rocm")]
        EpPreference::Rocm => {
            let ep = ort::ep::ROCm::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-openvino")]
        EpPreference::OpenVino => {
            let ep = ort::ep::OpenVINO::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-qnn")]
        EpPreference::Qnn => {
            let ep = ort::ep::QNN::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-nnapi")]
        EpPreference::Nnapi => {
            let ep = ort::ep::NNAPI::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(feature = "ort-xnnpack")]
        EpPreference::Xnnpack => {
            let ep = ort::ep::XNNPACK::default();
            (
                ep.is_available().unwrap_or(false),
                true,
                ep.name().to_string(),
            )
        }
        #[cfg(not(feature = "ort-cuda"))]
        EpPreference::Cuda => (false, false, "CUDAExecutionProvider".into()),
        #[cfg(not(feature = "ort-tensorrt"))]
        EpPreference::TensorRt => (false, false, "TensorrtExecutionProvider".into()),
        #[cfg(not(feature = "ort-rocm"))]
        EpPreference::Rocm => (false, false, "ROCMExecutionProvider".into()),
        #[cfg(not(feature = "ort-openvino"))]
        EpPreference::OpenVino => (false, false, "OpenVINOExecutionProvider".into()),
        #[cfg(not(feature = "ort-qnn"))]
        EpPreference::Qnn => (false, false, "QNNExecutionProvider".into()),
        #[cfg(not(feature = "ort-nnapi"))]
        EpPreference::Nnapi => (false, false, "NNAPIExecutionProvider".into()),
        #[cfg(not(feature = "ort-xnnpack"))]
        EpPreference::Xnnpack => (false, false, "XNNPACKExecutionProvider".into()),
    };
    ProviderInfo {
        name,
        preference: pref,
        available,
        platform_ok,
    }
}

/// Probe the default EP chain and print/log results.
pub fn probe_providers() -> Vec<ProviderInfo> {
    let mut out = Vec::new();
    for pref in EpPreference::default_chain() {
        let info = probe_one(pref);
        if info.available {
            info!(
                ep = %info.name,
                key = info.preference.as_str(),
                "execution provider available"
            );
        } else {
            debug!(
                ep = %info.name,
                key = info.preference.as_str(),
                platform_ok = info.platform_ok,
                "execution provider not available"
            );
        }
        out.push(info);
    }
    out
}

/// Build an ordered EP dispatch list from prefs, skipping unavailable ones.
/// Always ends with CPU as a safety net.
pub fn resolve_ep_chain(prefs: &[EpPreference]) -> Vec<ExecutionProviderDispatch> {
    let chain = if prefs.is_empty() {
        EpPreference::default_chain()
    } else {
        prefs.to_vec()
    };

    let mut selected = Vec::new();
    let mut saw_cpu = false;
    for pref in chain {
        let info = probe_one(pref);
        if !info.available {
            warn!(
                ep = pref.as_str(),
                "EP preferred but unavailable — skipping"
            );
            continue;
        }
        let Some(dispatch) = dispatch_for(pref) else {
            warn!(ep = pref.as_str(), "EP not compiled into this build — skipping");
            continue;
        };
        if pref == EpPreference::Cpu {
            saw_cpu = true;
        }
        debug!(ep = pref.as_str(), "selecting EP");
        selected.push(dispatch);
    }
    if !saw_cpu {
        selected.push(CPU::default().build());
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ep_preference_as_str() {
        assert_eq!(EpPreference::TensorRt.as_str(), "tensorrt");
        assert_eq!(EpPreference::Cuda.as_str(), "cuda");
        assert_eq!(EpPreference::Rocm.as_str(), "rocm");
        assert_eq!(EpPreference::OpenVino.as_str(), "openvino");
        assert_eq!(EpPreference::Qnn.as_str(), "qnn");
        assert_eq!(EpPreference::Nnapi.as_str(), "nnapi");
        assert_eq!(EpPreference::Xnnpack.as_str(), "xnnpack");
        assert_eq!(EpPreference::Cpu.as_str(), "cpu");
    }

    #[test]
    fn test_ep_preference_serde_json() {
        let qnn: EpPreference = serde_json::from_str("\"qnn\"").unwrap();
        assert_eq!(qnn, EpPreference::Qnn);

        let nnapi: EpPreference = serde_json::from_str("\"nnapi\"").unwrap();
        assert_eq!(nnapi, EpPreference::Nnapi);

        let xnnpack: EpPreference = serde_json::from_str("\"xnnpack\"").unwrap();
        assert_eq!(xnnpack, EpPreference::Xnnpack);

        let json = serde_json::to_string(&EpPreference::Qnn).unwrap();
        assert_eq!(json, "\"qnn\"");
    }

    #[test]
    fn test_default_chain_contents() {
        let chain = EpPreference::default_chain();
        assert!(chain.contains(&EpPreference::Qnn));
        assert!(chain.contains(&EpPreference::Nnapi));
        assert!(chain.contains(&EpPreference::Xnnpack));
        assert_eq!(chain.last(), Some(&EpPreference::Cpu));
    }

    #[test]
    fn test_probe_providers_has_cpu_available() {
        let providers = probe_providers();
        let cpu_info = providers.iter().find(|p| p.preference == EpPreference::Cpu);
        assert!(cpu_info.is_some());
        assert!(cpu_info.unwrap().available);
    }
}
