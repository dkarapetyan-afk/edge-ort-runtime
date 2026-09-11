//! CLI: listen --source mic|monitor|fake|file --profile default --tts off|on

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use edge_ort_runtime::audio::{FakeSource, PipeWireSource, SourceKind, WavFileSource};
use edge_ort_runtime::nodes::NodeRegistry;
use edge_ort_runtime::pipeline::{Pipeline, PipelineConfig};
use edge_ort_runtime::profile::Profile;
use edge_ort_runtime::runtime::probe_providers;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, ValueEnum)]
enum SourceArg {
    Mic,
    Monitor,
    Fake,
    File,
}

#[derive(Debug, Clone, ValueEnum)]
enum TtsArg {
    Off,
    On,
}

#[derive(Parser, Debug)]
#[command(
    name = "listen",
    about = "PipeWire audio → pluggable ONNX pipeline (VAD/ASR/MT/TTS)",
    long_about = None
)]
struct Cli {
    /// Audio source: mic, monitor (sink), fake (sine), or file (WAV)
    #[arg(long, value_enum, default_value_t = SourceArg::Fake)]
    source: SourceArg,

    /// WAV path when --source file (e.g. models/jfk.wav or tests/fixtures/jfk.wav)
    #[arg(long)]
    file: Option<PathBuf>,

    /// Profile name under profiles/<name>/profile.json, or a path to profile.json
    #[arg(long, default_value = "default")]
    profile: String,

    /// Enable TTS playback stub
    #[arg(long, value_enum, default_value_t = TtsArg::Off)]
    tts: TtsArg,

    /// Disable VAD gating
    #[arg(long, default_value_t = false)]
    no_vad: bool,

    /// Disable MT stage
    #[arg(long, default_value_t = false)]
    no_mt: bool,

    /// Limit number of audio chunks (useful for demos / CI)
    #[arg(long)]
    max_chunks: Option<usize>,

    /// Seconds of fake sine when --source fake (default 2)
    #[arg(long, default_value_t = 2.0)]
    fake_secs: f32,
}

fn resolve_profile(name_or_path: &str) -> Result<PathBuf> {
    let p = PathBuf::from(name_or_path);
    if p.is_file() {
        return Ok(p);
    }
    let candidates = [
        PathBuf::from(format!("profiles/{name_or_path}/profile.json")),
        PathBuf::from(format!("profiles/{name_or_path}.json")),
        PathBuf::from(name_or_path),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let c = PathBuf::from(manifest_dir)
            .join("profiles")
            .join(name_or_path)
            .join("profile.json");
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!(
        "profile not found: {name_or_path} (tried profiles/{name_or_path}/profile.json)"
    );
}

fn resolve_wav(cli: &Cli) -> Result<PathBuf> {
    if let Some(p) = &cli.file {
        if p.is_file() {
            return Ok(p.clone());
        }
        bail!("WAV not found: {}", p.display());
    }
    let candidates = [
        PathBuf::from("models/jfk.wav"),
        PathBuf::from("tests/fixtures/jfk.wav"),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        for rel in ["models/jfk.wav", "tests/fixtures/jfk.wav"] {
            let c = PathBuf::from(&manifest_dir).join(rel);
            if c.is_file() {
                return Ok(c);
            }
        }
    }
    bail!("--source file needs --file PATH (or models/jfk.wav from download-models.sh)");
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    tracing::info!("probing ONNX Runtime execution providers…");
    let providers = probe_providers();
    for p in &providers {
        tracing::info!(
            ep = %p.name,
            key = p.preference.as_str(),
            available = p.available,
            "EP"
        );
    }

    let profile_path = resolve_profile(&cli.profile)
        .with_context(|| format!("resolving profile '{}'", cli.profile))?;
    let profile = Profile::load(&profile_path)
        .with_context(|| format!("loading {}", profile_path.display()))?;
    tracing::info!(name = %profile.name, path = %profile_path.display(), "loaded profile");

    let manifests = profile
        .load_manifests(&profile_path)
        .context("loading manifests")?;
    let registry = NodeRegistry::from_manifests(manifests);

    let config = PipelineConfig {
        enable_vad: !cli.no_vad,
        enable_mt: !cli.no_mt,
        enable_tts: matches!(cli.tts, TtsArg::On),
        max_chunks: cli.max_chunks.or(match cli.source {
            SourceArg::Fake => Some(25),
            SourceArg::File => Some(8),
            _ => None,
        }),
        stop: None,
    };

    let mut pipeline = Pipeline::new(registry, config);

    match cli.source {
        SourceArg::Fake => {
            let mut src = FakeSource::sine(cli.fake_secs);
            pipeline.run(&mut src)?;
        }
        SourceArg::File => {
            let path = resolve_wav(&cli)?;
            tracing::info!(path = %path.display(), "WAV file source");
            let mut src = WavFileSource::open(&path)?;
            pipeline.run(&mut src)?;
        }
        SourceArg::Mic | SourceArg::Monitor => {
            let kind = match cli.source {
                SourceArg::Mic => SourceKind::Mic,
                SourceArg::Monitor => SourceKind::Monitor,
                _ => unreachable!(),
            };
            match PipeWireSource::open(kind) {
                Ok(mut src) => pipeline.run(&mut src)?,
                Err(e) => {
                    tracing::error!(error = %e, "PipeWire open failed — try --source fake or --source file");
                    bail!(e);
                }
            }
        }
    }

    Ok(())
}
