//! Tests for multilingual ASR, language matrices, and translation pipelines.

use edge_ort_runtime::audio::{AudioSource, FakeSource, WavFileSource};
use edge_ort_runtime::nodes::{AsrNode, MtNode, NodeRegistry};
use edge_ort_runtime::pipeline::{Pipeline, PipelineConfig, PipelineEvent};
use edge_ort_runtime::profile::{CapabilityKind, Manifest, SupportedLanguage};
use edge_ort_runtime::runtime::SessionManager;
use std::collections::HashMap;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ggml_model() -> PathBuf {
    repo_root().join("models/ggml-tiny.bin")
}

fn jfk_wav() -> PathBuf {
    let a = repo_root().join("tests/fixtures/jfk.wav");
    if a.is_file() {
        return a;
    }
    repo_root().join("models/jfk.wav")
}

// ----------------------------------------------------------------------------
// 1. Manifest and Profile Language Coverage
// ----------------------------------------------------------------------------

#[test]
fn test_default_asr_manifest_has_all_languages() {
    let manifest_path = repo_root().join("profiles/default/asr.whisper.json");
    let content = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));
    let manifest: Manifest = serde_json::from_str(&content).expect("parse manifest");

    assert_eq!(manifest.capability, CapabilityKind::Asr);

    // Verify all 15 explicit supported languages are listed in the manifest
    for lang in SupportedLanguage::ALL {
        if *lang == SupportedLanguage::Auto {
            continue;
        }
        let code = lang.code();
        assert!(
            manifest.languages.iter().any(|l| l == code),
            "manifest should list supported language: {code}"
        );
    }
    assert!(
        manifest.languages.iter().any(|l| l == "multilingual"),
        "manifest should include 'multilingual' tag"
    );
}

#[test]
fn test_default_mt_manifest_validity() {
    let manifest_path = repo_root().join("profiles/default/mt.nllb.json");
    let content = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));
    let manifest: Manifest = serde_json::from_str(&content).expect("parse manifest");

    assert_eq!(manifest.capability, CapabilityKind::Mt);
    assert!(!manifest.languages.is_empty());
    assert!(manifest.languages.contains(&"en".to_string()));
}

// ----------------------------------------------------------------------------
// 2. Exhaustive Language Configuration
// ----------------------------------------------------------------------------

#[test]
fn test_all_languages_configurable_on_asr_node() {
    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    let manifest = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: PathBuf::from("models/ggml-tiny.bin"),
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["multilingual".into()],
        ep_prefs: vec![],
        meta,
    };

    let mut asr = AsrNode::from_manifest(manifest);

    for lang in SupportedLanguage::ALL {
        let code = lang.code();
        asr.set_language(code);
        assert_eq!(asr.language(), code);

        let parsed = SupportedLanguage::from_code(code);
        if *lang == SupportedLanguage::ZhTw {
            assert_eq!(parsed, Some(SupportedLanguage::Zh));
        } else {
            assert_eq!(
                parsed,
                Some(*lang),
                "SupportedLanguage::from_code should round-trip for {code}"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Pipeline Translation Routing & Dual-Translate
// ----------------------------------------------------------------------------

#[test]
fn test_pipeline_enables_dual_translate_when_mt_active() {
    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    let asr_man = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: PathBuf::from("models/ggml-tiny.bin"),
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["fr".into()],
        ep_prefs: vec![],
        meta,
    };

    let asr = AsrNode::from_manifest(asr_man);
    let registry = NodeRegistry {
        asr: Some(asr),
        ..Default::default()
    };

    let config = PipelineConfig {
        enable_vad: false,
        enable_mt: true,
        enable_tts: false,
        max_chunks: Some(1),
        stop: None,
    };

    let mut pipeline = Pipeline::new(registry, config);
    let mut src = FakeSource::sine(0.1);

    let mut events = Vec::new();
    pipeline
        .run_with_handler(&mut src, |ev| {
            events.push(ev);
            true
        })
        .expect("run pipeline");

    let asr_node = pipeline.registry.asr.as_ref().expect("asr node");
    assert!(
        asr_node.dual_translate(),
        "pipeline with enable_mt=true must enable dual_translate on ASR"
    );
}

#[test]
fn test_pipeline_suppresses_dual_translate_when_mt_disabled() {
    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    let asr_man = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: PathBuf::from("models/ggml-tiny.bin"),
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["fr".into()],
        ep_prefs: vec![],
        meta,
    };

    let asr = AsrNode::from_manifest(asr_man);
    let registry = NodeRegistry {
        asr: Some(asr),
        ..Default::default()
    };

    let config = PipelineConfig {
        enable_vad: false,
        enable_mt: false,
        enable_tts: false,
        max_chunks: Some(1),
        stop: None,
    };

    let mut pipeline = Pipeline::new(registry, config);
    let mut src = FakeSource::sine(0.1);

    pipeline
        .run_with_handler(&mut src, |_| true)
        .expect("run pipeline");

    let asr_node = pipeline.registry.asr.as_ref().expect("asr node");
    assert!(
        !asr_node.dual_translate(),
        "pipeline with enable_mt=false must not enable dual_translate on ASR"
    );
}

#[test]
fn test_pipeline_fallback_to_mt_node_when_asr_translation_none() {
    // Construct ASR node using tiny_energy (which emits translation: None)
    let asr_man = Manifest {
        id: "tiny-energy".into(),
        capability: CapabilityKind::Asr,
        model_path: repo_root().join("models/tiny_asr_energy.onnx"),
        inputs: vec!["audio".into()],
        outputs: vec!["logits".into()],
        sample_rate: 16_000,
        languages: vec!["en".into()],
        ep_prefs: vec![],
        meta: HashMap::new(),
    };

    let mt_man = Manifest {
        id: "nllb-mt".into(),
        capability: CapabilityKind::Mt,
        model_path: PathBuf::from("models/nllb.onnx"),
        inputs: vec!["input_ids".into()],
        outputs: vec!["logits".into()],
        sample_rate: 16_000,
        languages: vec!["es".into(), "en".into()],
        ep_prefs: vec![],
        meta: HashMap::new(),
    };

    let asr = AsrNode::from_manifest(asr_man);
    let mt = MtNode::from_manifest(mt_man);
    let registry = NodeRegistry {
        asr: Some(asr),
        mt: Some(mt),
        ..Default::default()
    };

    let config = PipelineConfig {
        enable_vad: false,
        enable_mt: true,
        enable_tts: false,
        max_chunks: Some(1),
        stop: None,
    };

    let mut pipeline = Pipeline::new(registry, config);
    let mut src = FakeSource::sine(0.2);

    let mut transcripts = Vec::new();
    pipeline
        .run_with_handler(&mut src, |ev| {
            if let PipelineEvent::Transcript { text, translation, .. } = ev {
                transcripts.push((text, translation));
            }
            true
        })
        .expect("run pipeline");

    // If stub transcript was emitted, mt.process should have populated translation
    for (text, translation) in transcripts {
        if !text.is_empty() {
            assert!(
                translation.is_some(),
                "when enable_mt=true and MT node is present, translation should be populated"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// 4. Live Whisper GGML Smoke Tests (Skipped if model absent)
// ----------------------------------------------------------------------------

#[test]
fn test_whisper_ggml_transcribe_en_jfk() {
    let model = ggml_model();
    let wav = jfk_wav();
    if !model.is_file() || !wav.is_file() {
        eprintln!("skip live test: missing model or WAV fixture");
        return;
    }

    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    meta.insert("language".into(), serde_json::json!("en"));
    meta.insert("task".into(), serde_json::json!("transcribe"));
    meta.insert("n_threads".into(), serde_json::json!(2));

    let manifest = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: model,
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["en".into()],
        ep_prefs: vec![],
        meta,
    };

    let mut asr = AsrNode::from_manifest(manifest);
    let mut sessions = SessionManager::new();
    let mut src = WavFileSource::open(&wav).expect("open wav");
    let pcm = src.pull().expect("pull");

    let out = asr.process(&mut sessions, &pcm).expect("asr");
    assert!(!out.text.is_empty());
    assert!(out.confidence > 0.0);
    assert_eq!(out.translation, None, "transcribe mode should not populate translation");

    let lower = out.text.to_ascii_lowercase();
    assert!(
        lower.contains("fellow") || lower.contains("citizen") || lower.contains("ask") || lower.contains("country"),
        "recognized text should contain JFK keywords: {lower}"
    );
}

#[test]
fn test_whisper_ggml_auto_language_detect_jfk() {
    let model = ggml_model();
    let wav = jfk_wav();
    if !model.is_file() || !wav.is_file() {
        eprintln!("skip live test: missing model or WAV fixture");
        return;
    }

    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    meta.insert("language".into(), serde_json::json!("auto"));
    meta.insert("task".into(), serde_json::json!("transcribe"));
    meta.insert("n_threads".into(), serde_json::json!(2));

    let manifest = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: model,
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["multilingual".into()],
        ep_prefs: vec![],
        meta,
    };

    let mut asr = AsrNode::from_manifest(manifest);
    let mut sessions = SessionManager::new();
    let mut src = WavFileSource::open(&wav).expect("open wav");
    let pcm = src.pull().expect("pull");

    let out = asr.process(&mut sessions, &pcm).expect("asr auto detect");
    assert!(!out.text.is_empty());
    let lower = out.text.to_ascii_lowercase();
    assert!(
        lower.contains("country") || lower.contains("ask") || lower.contains("citizen"),
        "auto-detected language should transcribe English: {lower}"
    );
}

#[test]
fn test_whisper_ggml_dual_translate_en_skips_redundant_translation() {
    let model = ggml_model();
    let wav = jfk_wav();
    if !model.is_file() || !wav.is_file() {
        eprintln!("skip live test: missing model or WAV fixture");
        return;
    }

    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    meta.insert("language".into(), serde_json::json!("en"));
    meta.insert("task".into(), serde_json::json!("transcribe"));
    meta.insert("n_threads".into(), serde_json::json!(2));

    let manifest = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: model,
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["en".into()],
        ep_prefs: vec![],
        meta,
    };

    let mut asr = AsrNode::from_manifest(manifest);
    asr.set_dual_translate(true);

    let mut sessions = SessionManager::new();
    let mut src = WavFileSource::open(&wav).expect("open wav");
    let pcm = src.pull().expect("pull");

    let out = asr.process(&mut sessions, &pcm).expect("asr dual translate");
    assert!(!out.text.is_empty());
    // For English, translation is left as None because original is already English
    assert_eq!(
        out.translation, None,
        "English source audio in dual-translate mode should leave translation as None"
    );
}

#[test]
fn test_whisper_ggml_translate_task_flag() {
    let model = ggml_model();
    let wav = jfk_wav();
    if !model.is_file() || !wav.is_file() {
        eprintln!("skip live test: missing model or WAV fixture");
        return;
    }

    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    meta.insert("language".into(), serde_json::json!("en"));
    meta.insert("task".into(), serde_json::json!("translate"));
    meta.insert("n_threads".into(), serde_json::json!(2));

    let manifest = Manifest {
        id: "whisper-test".into(),
        capability: CapabilityKind::Asr,
        model_path: model,
        inputs: vec!["pcm".into()],
        outputs: vec!["text".into()],
        sample_rate: 16_000,
        languages: vec!["en".into()],
        ep_prefs: vec![],
        meta,
    };

    let mut asr = AsrNode::from_manifest(manifest);
    let mut sessions = SessionManager::new();
    let mut src = WavFileSource::open(&wav).expect("open wav");
    let pcm = src.pull().expect("pull");

    let out = asr.process(&mut sessions, &pcm).expect("asr translate task");
    assert!(!out.text.is_empty());
}
