//! Smoke: Whisper ggml ASR on a real speech WAV yields English words (not stubs).

use edge_ort_runtime::audio::{AudioSource, WavFileSource};
use edge_ort_runtime::nodes::AsrNode;
use edge_ort_runtime::profile::{CapabilityKind, Manifest, Profile};
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

#[test]
fn whisper_ggml_transcribes_jfk() {
    let model = ggml_model();
    let wav = jfk_wav();
    if !model.is_file() {
        eprintln!(
            "skip: missing {} (run ./scripts/download-models.sh)",
            model.display()
        );
        return;
    }
    if !wav.is_file() {
        eprintln!("skip: missing {}", wav.display());
        return;
    }

    let mut meta = HashMap::new();
    meta.insert("backend".into(), serde_json::json!("whisper_ggml"));
    meta.insert("language".into(), serde_json::json!("en"));
    meta.insert("task".into(), serde_json::json!("transcribe"));
    meta.insert("n_threads".into(), serde_json::json!(2));

    let manifest = Manifest {
        id: "whisper-tiny".into(),
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
    assert!(
        pcm.len() > 16_000,
        "expected >1s of audio, got {}",
        pcm.len()
    );

    let out = asr.process(&mut sessions, &pcm).expect("asr");
    eprintln!("transcript: {:?}", out.text);
    let lower = out.text.to_ascii_lowercase();
    assert!(
        !lower.contains("[utterance]")
            && !lower.contains("[stub")
            && !lower.contains("todo")
            && !lower.is_empty(),
        "expected real English words, got {:?}",
        out.text
    );
    // JFK sample classic line — tolerate tiny-model fuzz
    let hit = ["ask", "country", "able", "do", "for", "you", "fellow", "citizen"]
        .iter()
        .any(|w| lower.contains(w));
    assert!(
        hit,
        "transcript should resemble JFK speech, got {:?}",
        out.text
    );
}

#[test]
fn default_profile_points_at_whisper() {
    let profile_path = repo_root().join("profiles/default/profile.json");
    let profile = Profile::load(&profile_path).expect("profile");
    let asr_rel = profile.nodes.get("asr").expect("asr key");
    assert!(
        asr_rel.to_string_lossy().contains("whisper"),
        "default ASR should be whisper plug, got {}",
        asr_rel.display()
    );
    let manifests = profile.load_manifests(&profile_path).expect("manifests");
    let asr = manifests.get(&CapabilityKind::Asr).expect("asr man");
    let backend = asr
        .meta
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(backend, "whisper_ggml");
}
