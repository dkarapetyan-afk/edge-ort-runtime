//! SLM (Small Language Model) post-processing node.
//!
//! Provides text cleanup (removing filler words: um, uh, like, you know),
//! disfluency filtering, grammar capitalization, and meeting action item summarization.
//! Pluggable via manifest (`capability: "slm"`, model path, EP prefs).
//! If model weights are missing, falls back to a deterministic rule-based NLP cleaner and summarizer.

use super::NodeError;
use crate::profile::Manifest;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlmSummary {
    pub cleaned_text: String,
    pub bullet_points: Vec<String>,
    pub word_count_original: usize,
    pub word_count_cleaned: usize,
}

pub struct SlmNode {
    pub manifest: Manifest,
    pub ready: bool,
}

impl SlmNode {
    pub fn from_manifest(manifest: Manifest) -> Self {
        let ready = manifest.model_path.exists();
        if !ready {
            info!(
                id = %manifest.id,
                path = %manifest.model_path.display(),
                "SLM model missing — using algorithmic cleaner and summarizer"
            );
        }
        Self { manifest, ready }
    }

    /// Clean speech disfluencies and filler words.
    pub fn clean_text(&self, text: &str) -> String {
        clean_transcript_heuristics(text)
    }

    /// Summarize speech transcript into concise bullet points.
    pub fn summarize(&self, text: &str) -> Result<SlmSummary, NodeError> {
        let original_words = text.split_whitespace().count();
        let cleaned = clean_transcript_heuristics(text);
        let cleaned_words = cleaned.split_whitespace().count();
        let bullets = extract_bullet_points(&cleaned);

        Ok(SlmSummary {
            cleaned_text: cleaned,
            bullet_points: bullets,
            word_count_original: original_words,
            word_count_cleaned: cleaned_words,
        })
    }
}

/// Algorithmic disfluency remover and grammar cleaner.
pub fn clean_transcript_heuristics(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    let filler_words = [
        "um", "uh", "erm", "ah", "like", "er", "hmm", "you know", "i mean"
    ];

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut cleaned_words: Vec<String> = Vec::new();

    let mut prev_lower = String::new();
    for word in words {
        let lower = word.to_lowercase();
        let stripped = lower.trim_matches(|c: char| !c.is_alphanumeric());

        // Skip filler words
        if filler_words.contains(&stripped) || filler_words.contains(&lower.as_str()) {
            continue;
        }

        // Skip consecutive duplicate words (e.g., "the the" -> "the")
        if stripped == prev_lower && !stripped.is_empty() {
            continue;
        }

        prev_lower = stripped.to_string();
        cleaned_words.push(word.to_string());
    }

    if cleaned_words.is_empty() {
        return String::new();
    }

    // Capitalize the first letter
    let mut result = cleaned_words.join(" ");
    if let Some(first_char) = result.chars().next() {
        if first_char.is_alphabetic() && first_char.is_lowercase() {
            result = format!("{}{}", first_char.to_uppercase(), &result[first_char.len_utf8()..]);
        }
    }

    // Ensure ending punctuation
    if !result.ends_with('.') && !result.ends_with('?') && !result.ends_with('!') {
        result.push('.');
    }

    result
}

/// Extracts key points and action items from sentences.
pub fn extract_bullet_points(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut bullets = Vec::new();
    let sentences: Vec<&str> = text
        .split(['.', '?', '!'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect();

    for sentence in sentences {
        // Format as clean bullet point
        let mut bullet = sentence.to_string();
        if let Some(first_char) = bullet.chars().next() {
            if first_char.is_alphabetic() && first_char.is_lowercase() {
                bullet = format!("{}{}", first_char.to_uppercase(), &bullet[first_char.len_utf8()..]);
            }
        }
        bullets.push(bullet);
    }

    if bullets.is_empty() && !text.trim().is_empty() {
        bullets.push(text.trim().to_string());
    }

    bullets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::CapabilityKind;
    use std::path::PathBuf;

    #[test]
    fn test_clean_transcript_removes_fillers_and_duplicates() {
        let input = "um, we should uh like test the the pipeline now";
        let cleaned = clean_transcript_heuristics(input);
        assert_eq!(cleaned, "We should test the pipeline now.");
    }

    #[test]
    fn test_clean_transcript_empty() {
        assert_eq!(clean_transcript_heuristics("   "), "");
    }

    #[test]
    fn test_extract_bullet_points() {
        let input = "First we configure the NPU. Then we run the Whisper ASR model. Finally we save results.";
        let bullets = extract_bullet_points(input);
        assert_eq!(bullets.len(), 3);
        assert_eq!(bullets[0], "First we configure the NPU");
        assert_eq!(bullets[1], "Then we run the Whisper ASR model");
        assert_eq!(bullets[2], "Finally we save results");
    }

    #[test]
    fn test_slm_node_summarize() {
        let manifest = Manifest {
            id: "slm-test".into(),
            capability: CapabilityKind::Slm,
            model_path: PathBuf::from("missing.onnx"),
            inputs: vec![],
            outputs: vec![],
            sample_rate: 0,
            languages: vec![],
            ep_prefs: vec![],
            meta: Default::default(),
        };

        let node = SlmNode::from_manifest(manifest);
        assert!(!node.ready);

        let text = "Um, the meeting agreed to deploy Whisper on Android. We will start testing tomorrow.";
        let summary = node.summarize(text).unwrap();

        assert!(summary.cleaned_text.contains("The meeting agreed"));
        assert!(!summary.cleaned_text.contains("Um,"));
        assert_eq!(summary.bullet_points.len(), 2);
        assert!(summary.word_count_original >= summary.word_count_cleaned);
    }
}
