//! Execution Provider priority chain with graceful degradation.

use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    OpenVINOExecutionProvider, ROCmExecutionProvider, TensorRTExecutionProvider,
};
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
    Cpu,
}

impl EpPreference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TensorRt => "tensorrt",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::OpenVino => "openvino",
            Self::Cpu => "cpu",
        }
    }

    /// Default priority: TensorRT → CUDA → ROCm → OpenVINO → CPU.
    pub fn default_chain() -> Vec<Self> {
        vec![
            Self::TensorRt,
            Self::Cuda,
            Self::Rocm,
            Self::OpenVino,
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

fn dispatch_for(pref: EpPreference) -> ExecutionProviderDispatch {
    match pref {
        EpPreference::TensorRt => TensorRTExecutionProvider::default().build(),
        EpPreference::Cuda => CUDAExecutionProvider::default().build(),
        EpPreference::Rocm => ROCmExecutionProvider::default().build(),
        EpPreference::OpenVino => OpenVINOExecutionProvider::default().build(),
        EpPreference::Cpu => CPUExecutionProvider::default().build(),
    }
}

fn probe_one(pref: EpPreference) -> ProviderInfo {
    // Build a temporary EP just to query availability.
    let (available, platform_ok, name) = match pref {
        EpPreference::TensorRt => {
            let ep = TensorRTExecutionProvider::default();
            (
                ep.is_available().unwrap_or(false),
                ep.supported_by_platform(),
                ep.name().to_string(),
            )
        }
        EpPreference::Cuda => {
            let ep = CUDAExecutionProvider::default();
            (
                ep.is_available().unwrap_or(false),
                ep.supported_by_platform(),
                ep.name().to_string(),
            )
        }
        EpPreference::Rocm => {
            let ep = ROCmExecutionProvider::default();
            (
                ep.is_available().unwrap_or(false),
                ep.supported_by_platform(),
                ep.name().to_string(),
            )
        }
        EpPreference::OpenVino => {
            let ep = OpenVINOExecutionProvider::default();
            (
                ep.is_available().unwrap_or(false),
                ep.supported_by_platform(),
                ep.name().to_string(),
            )
        }
        EpPreference::Cpu => {
            let ep = CPUExecutionProvider::default();
            (
                ep.is_available().unwrap_or(false),
                ep.supported_by_platform(),
                ep.name().to_string(),
            )
        }
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
        if pref == EpPreference::Cpu {
            saw_cpu = true;
        }
        debug!(ep = pref.as_str(), "selecting EP");
        selected.push(dispatch_for(pref));
    }
    if !saw_cpu {
        selected.push(CPUExecutionProvider::default().build());
    }
    selected
}
