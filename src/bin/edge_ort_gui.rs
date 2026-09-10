//! Native GUI: source / profile / Start-Stop / live transcript / EP status / TTS.

use eframe::egui;
use edge_ort_runtime::audio::{FakeSource, PipeWireSource, SourceKind};
use edge_ort_runtime::nodes::NodeRegistry;
use edge_ort_runtime::pipeline::{Pipeline, PipelineConfig, PipelineEvent};
use edge_ort_runtime::profile::Profile;
use edge_ort_runtime::runtime::probe_providers;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tracing_subscriber::EnvFilter;

const PLACEHOLDER_ORIGINAL: &str = "مرحبا · 你好 · こんにちは · 안녕하세요 · नमस्ते · Привет";
const PLACEHOLDER_ENGLISH: &str = "Hello — font check";

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceChoice {
    Fake,
    Mic,
    Monitor,
}

impl SourceChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Fake => "fake",
            Self::Mic => "mic",
            Self::Monitor => "monitor",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LanguageChoice {
    Auto,
    En,
    Fr,
    Es,
    Ar,
    Ru,
    Hi,
    Zh,
    ZhTw,
    Ko,
    Ja,
    Ur,
    Fa,
    Tr,
    Az,
    Hy,
}

impl LanguageChoice {
    fn code(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::En => "en",
            Self::Fr => "fr",
            Self::Es => "es",
            Self::Ar => "ar",
            Self::Ru => "ru",
            Self::Hi => "hi",
            // Whisper has no separate Taiwanese/Hokkien code; both map to zh.
            Self::Zh | Self::ZhTw => "zh",
            Self::Ko => "ko",
            Self::Ja => "ja",
            Self::Ur => "ur",
            Self::Fa => "fa",
            Self::Tr => "tr",
            Self::Az => "az",
            Self::Hy => "hy",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::En => "English",
            Self::Fr => "French",
            Self::Es => "Spanish",
            Self::Ar => "Arabic",
            Self::Ru => "Russian",
            Self::Hi => "Hindi",
            Self::Zh => "Mandarin (zh)",
            Self::ZhTw => "Taiwanese (zh)",
            Self::Ko => "Korean",
            Self::Ja => "Japanese",
            Self::Ur => "Urdu (PK)",
            Self::Fa => "Farsi",
            Self::Tr => "Turkish",
            Self::Az => "Azerbaijani",
            Self::Hy => "Armenian",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WhisperSize {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3,
}

impl WhisperSize {
    fn label(self) -> &'static str {
        match self {
            Self::Tiny => "tiny (~75MB)",
            Self::Base => "base (~142MB)",
            Self::Small => "small (~466MB)",
            Self::Medium => "medium (~1.5GB)",
            Self::LargeV3 => "large-v3 (huge)",
        }
    }

    fn filename(self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
            Self::LargeV3 => "ggml-large-v3.bin",
        }
    }

    fn path(self) -> PathBuf {
        let name = self.filename();
        let candidates = [
            PathBuf::from("models").join(name),
            PathBuf::from(name),
        ];
        for c in &candidates {
            if c.is_file() {
                return c.clone();
            }
        }
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let c = PathBuf::from(manifest_dir).join("models").join(name);
            if c.is_file() {
                return c;
            }
        }
        let mut cur = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        for _ in 0..6 {
            let c = cur.join("models").join(name);
            if c.is_file() {
                return c;
            }
            if !cur.pop() {
                break;
            }
        }
        // Prefer models/ relative to cwd even if missing (caller checks exists).
        PathBuf::from("models").join(name)
    }

    fn exists(self) -> bool {
        self.path().is_file()
    }

    const ALL: [WhisperSize; 5] = [
        Self::Tiny,
        Self::Base,
        Self::Small,
        Self::Medium,
        Self::LargeV3,
    ];
}

enum WorkerMsg {
    Event(PipelineEvent),
    Finished(Result<(), String>),
}

struct GuiApp {
    source: SourceChoice,
    language: LanguageChoice,
    whisper_size: WhisperSize,
    profile: String,
    tts_on: bool,
    enable_vad: bool,
    enable_mt: bool,
    running: bool,
    stop_flag: Option<Arc<AtomicBool>>,
    worker: Option<JoinHandle<()>>,
    rx: Option<Receiver<WorkerMsg>>,
    transcript: String,
    translation: String,
    /// True while Original/English still show idle font-check placeholders.
    placeholders: bool,
    status: String,
    error: String,
    ep_lines: Vec<String>,
    last_vad: String,
}

impl Default for GuiApp {
    fn default() -> Self {
        let providers = probe_providers();
        let ep_lines = providers
            .iter()
            .map(|p| {
                format!(
                    "{}: {}",
                    p.preference.as_str(),
                    if p.available {
                        "available"
                    } else {
                        "n/a"
                    }
                )
            })
            .collect();
        Self {
            source: SourceChoice::Fake,
            language: LanguageChoice::Auto,
            whisper_size: WhisperSize::Tiny,
            profile: "default".into(),
            tts_on: false,
            enable_vad: true,
            enable_mt: true,
            running: false,
            stop_flag: None,
            worker: None,
            rx: None,
            transcript: PLACEHOLDER_ORIGINAL.into(),
            translation: PLACEHOLDER_ENGLISH.into(),
            placeholders: true,
            status: "idle".into(),
            error: String::new(),
            ep_lines,
            last_vad: "—".into(),
        }
    }
}

/// Resolve a font under assets/ relative to crate root / cwd / parents.
fn asset_font(rel: &str) -> Option<PathBuf> {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let c = PathBuf::from(manifest_dir).join(rel);
        if c.is_file() {
            return Some(c);
        }
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let c = cwd.join(rel);
    if c.is_file() {
        return Some(c);
    }
    let mut cur = cwd;
    for _ in 0..6 {
        let c = cur.join(rel);
        if c.is_file() {
            return Some(c);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

fn try_install_font(
    fonts: &mut egui::FontDefinitions,
    loaded: &mut Vec<&'static str>,
    name: &'static str,
    path: &Path,
) {
    match std::fs::read(path) {
        Ok(bytes) => {
            fonts
                .font_data
                .insert(name.to_string(), Arc::new(egui::FontData::from_owned(bytes)));
            loaded.push(name);
            tracing::info!(font = %name, path = %path.display(), "installed GUI font");
        }
        Err(e) => {
            eprintln!("[fonts] missing/unreadable {}: {e}", path.display());
            tracing::warn!(
                font = %name,
                path = %path.display(),
                error = %e,
                "GUI font unavailable"
            );
        }
    }
}

/// Load static TTFs (Arabic/Devanagari/CJK/Cyrillic) into egui (soft-fail per file).
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut loaded: Vec<&'static str> = Vec::new();

    let system: &[(&str, &str)] = &[
        (
            "dejavu",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ),
        (
            "dejavu_mono",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ),
        (
            "harmattan",
            "/usr/share/fonts/truetype/sand-box/google/Harmattan/Harmattan-Regular.ttf",
        ),
        (
            "martel",
            "/usr/share/fonts/truetype/sand-box/google/Martel/Martel-Regular.ttf",
        ),
        (
            "mirza",
            "/usr/share/fonts/truetype/sand-box/google/Mirza/Mirza-Regular.ttf",
        ),
    ];
    for (name, path) in system {
        try_install_font(&mut fonts, &mut loaded, name, Path::new(path));
    }

    let assets: &[(&str, &str)] = &[
        ("cjk_sc", "assets/fonts/NotoSansCJKsc-Regular.ttf"),
        ("cjk_tc", "assets/fonts/NotoSansCJKtc-Regular.ttf"),
        ("cjk_jp", "assets/fonts/NotoSansCJKjp-Regular.ttf"),
        ("cjk_kr", "assets/fonts/NotoSansCJKkr-Regular.ttf"),
    ];
    for (name, rel) in assets {
        match asset_font(rel) {
            Some(path) => try_install_font(&mut fonts, &mut loaded, name, &path),
            None => {
                eprintln!("[fonts] asset not found: {rel}");
                tracing::warn!(font = %name, path = rel, "GUI font asset not found");
            }
        }
    }

    // Preferred first: dejavu, harmattan, martel, then CJK faces.
    let prop_order: &[&str] = &[
        "dejavu",
        "harmattan",
        "mirza",
        "martel",
        "cjk_sc",
        "cjk_tc",
        "cjk_jp",
        "cjk_kr",
    ];
    if let Some(prop) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        for name in prop_order.iter().rev() {
            if loaded.contains(name) {
                prop.insert(0, (*name).to_owned());
            }
        }
    }

    // Monospace: dejavu_mono first; same proportional fallbacks after.
    if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        if loaded.contains(&"dejavu_mono") {
            mono.insert(0, "dejavu_mono".to_owned());
        }
        for name in prop_order {
            if loaded.contains(name) {
                mono.push((*name).to_owned());
            }
        }
    }

    ctx.set_fonts(fonts);
}

impl GuiApp {
    fn clear_placeholders(&mut self) {
        if self.placeholders {
            self.transcript.clear();
            self.translation.clear();
            self.placeholders = false;
        }
    }

    fn resolve_profile(name_or_path: &str) -> Result<PathBuf, String> {
        let p = PathBuf::from(name_or_path);
        if p.is_file() {
            return Ok(p);
        }
        let candidates = [
            PathBuf::from(format!("profiles/{name_or_path}/profile.json")),
            PathBuf::from(format!("profiles/{name_or_path}.json")),
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
        // Walk up from cwd looking for profiles/
        let mut cur = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        for _ in 0..6 {
            let c = cur.join("profiles").join(name_or_path).join("profile.json");
            if c.is_file() {
                return Ok(c);
            }
            if !cur.pop() {
                break;
            }
        }
        Err(format!(
            "profile not found: {name_or_path} (tried profiles/{name_or_path}/profile.json)"
        ))
    }

    fn start(&mut self) {
        if self.running {
            return;
        }
        self.error.clear();
        self.clear_placeholders();
        let profile_name = self.profile.clone();
        let profile_path = match Self::resolve_profile(&profile_name) {
            Ok(p) => p,
            Err(e) => {
                self.error = e;
                return;
            }
        };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);
        let (tx, rx) = mpsc::channel::<WorkerMsg>();
        let source = self.source;
        let language = self.language;
        let whisper_size = self.whisper_size;
        let tts_on = self.tts_on;
        let enable_vad = self.enable_vad;
        let enable_mt = self.enable_mt;

        let handle = thread::Builder::new()
            .name("pipeline".into())
            .spawn(move || {
                worker_main(
                    tx,
                    stop_worker,
                    source,
                    language,
                    whisper_size,
                    profile_path,
                    tts_on,
                    enable_vad,
                    enable_mt,
                );
            })
            .expect("spawn pipeline thread");

        self.stop_flag = Some(stop);
        self.worker = Some(handle);
        self.rx = Some(rx);
        self.running = true;
        self.status = format!("running ({})", whisper_size.filename());
    }

    fn stop(&mut self) {
        if let Some(flag) = &self.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
        self.status = "stopping…".into();
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.rx else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Event(PipelineEvent::Transcript {
                    text,
                    translation,
                    confidence,
                })) => {
                    if self.placeholders {
                        self.transcript.clear();
                        self.translation.clear();
                        self.placeholders = false;
                    }
                    if !self.transcript.is_empty() {
                        self.transcript.push('\n');
                    }
                    self.transcript
                        .push_str(&format!("{text}  (conf={confidence:.2})"));
                    if !self.translation.is_empty() {
                        self.translation.push('\n');
                    }
                    match translation {
                        Some(tr) if !tr.is_empty() => {
                            self.translation
                                .push_str(&format!("{tr}  (conf={confidence:.2})"));
                        }
                        _ => {
                            self.translation.push_str("—");
                        }
                    }
                }
                Ok(WorkerMsg::Event(PipelineEvent::Vad {
                    speech_prob,
                    is_speech,
                })) => {
                    self.last_vad = format!(
                        "prob={speech_prob:.2} {}",
                        if is_speech { "SPEECH" } else { "silence" }
                    );
                }
                Ok(WorkerMsg::Event(PipelineEvent::Status(s))) => {
                    self.status = s;
                }
                Ok(WorkerMsg::Event(PipelineEvent::Error(e))) => {
                    self.error = e;
                }
                Ok(WorkerMsg::Event(PipelineEvent::EpInfo { name, available })) => {
                    self.ep_lines.push(format!(
                        "{name}: {}",
                        if available { "available" } else { "n/a" }
                    ));
                }
                Ok(WorkerMsg::Event(PipelineEvent::Done)) => {
                    self.status = "done".into();
                }
                Ok(WorkerMsg::Finished(res)) => {
                    if let Err(e) = res {
                        self.error = e;
                        self.status = "error".into();
                    } else if self.status == "stopping…" || self.status.starts_with("running") {
                        self.status = "stopped".into();
                    }
                    self.running = false;
                    self.stop_flag = None;
                    if let Some(h) = self.worker.take() {
                        let _ = h.join();
                    }
                    self.rx = None;
                    return;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.running = false;
                    self.rx = None;
                    self.status = "worker disconnected".into();
                    break;
                }
            }
        }
    }
}

fn worker_main(
    tx: Sender<WorkerMsg>,
    stop: Arc<AtomicBool>,
    source: SourceChoice,
    language: LanguageChoice,
    whisper_size: WhisperSize,
    profile_path: PathBuf,
    tts_on: bool,
    enable_vad: bool,
    enable_mt: bool,
) {
    let send_ev = |tx: &Sender<WorkerMsg>, ev: PipelineEvent| -> bool {
        tx.send(WorkerMsg::Event(ev)).is_ok() && !stop.load(Ordering::Relaxed)
    };

    let result = (|| -> Result<(), String> {
        let profile = Profile::load(&profile_path).map_err(|e| e.to_string())?;
        let manifests = profile
            .load_manifests(&profile_path)
            .map_err(|e| e.to_string())?;
        let mut registry = NodeRegistry::from_manifests(manifests);
        if let Some(asr) = registry.asr.as_mut() {
            asr.set_language(language.code());
            let model_path = whisper_size.path();
            if !model_path.is_file() {
                let msg = format!(
                    "Whisper model missing: {} — run scripts/download-models.sh (WHISPER_SIZES includes {})",
                    model_path.display(),
                    match whisper_size {
                        WhisperSize::Tiny => "tiny",
                        WhisperSize::Base => "base",
                        WhisperSize::Small => "small",
                        WhisperSize::Medium => "medium",
                        WhisperSize::LargeV3 => "large-v3",
                    }
                );
                let _ = send_ev(&tx, PipelineEvent::Error(msg.clone()));
                return Err(msg);
            }
            asr.set_model_path(model_path.clone());
            let _ = send_ev(
                &tx,
                PipelineEvent::Status(format!(
                    "asr language={} model={}",
                    language.code(),
                    Path::new(whisper_size.filename()).display()
                )),
            );
        }

        let config = PipelineConfig {
            enable_vad,
            enable_mt,
            enable_tts: tts_on,
            max_chunks: None,
            stop: Some(Arc::clone(&stop)),
        };
        let mut pipeline = Pipeline::new(registry, config);

        let mut handler = |ev: PipelineEvent| send_ev(&tx, ev);

        match source {
            SourceChoice::Fake => {
                let mut src = FakeSource::sine_forever();
                pipeline
                    .run_with_handler(&mut src, &mut handler)
                    .map_err(|e| e.to_string())?;
            }
            SourceChoice::Mic | SourceChoice::Monitor => {
                let kind = match source {
                    SourceChoice::Mic => SourceKind::Mic,
                    SourceChoice::Monitor => SourceKind::Monitor,
                    SourceChoice::Fake => unreachable!(),
                };
                match PipeWireSource::open(kind) {
                    Ok(mut src) => {
                        pipeline
                            .run_with_handler(&mut src, &mut handler)
                            .map_err(|e| e.to_string())?;
                    }
                    Err(e) => {
                        let msg = format!("{e} — try source=fake");
                        let _ = send_ev(&tx, PipelineEvent::Error(msg.clone()));
                        return Err(msg);
                    }
                }
            }
        }
        Ok(())
    })();

    let _ = tx.send(WorkerMsg::Finished(result));
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        if self.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("edge-ort-runtime");
                ui.separator();
                ui.label(format!("status: {}", self.status));
            });
        });

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Controls");
                ui.add_space(8.0);

                ui.label("Source");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.source, SourceChoice::Fake, "fake");
                    ui.selectable_value(&mut self.source, SourceChoice::Mic, "mic");
                    ui.selectable_value(&mut self.source, SourceChoice::Monitor, "monitor");
                });
                ui.label(format!("selected: {}", self.source.label()));

                ui.add_space(8.0);
                ui.label("Profile");
                ui.text_edit_singleline(&mut self.profile);

                ui.add_space(8.0);
                ui.label("Language");
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.language, LanguageChoice::Auto, "auto");
                    ui.selectable_value(&mut self.language, LanguageChoice::En, "EN");
                    ui.selectable_value(&mut self.language, LanguageChoice::Fr, "FR");
                    ui.selectable_value(&mut self.language, LanguageChoice::Es, "ES");
                    ui.selectable_value(&mut self.language, LanguageChoice::Ar, "AR");
                    ui.selectable_value(&mut self.language, LanguageChoice::Ru, "RU");
                    ui.selectable_value(&mut self.language, LanguageChoice::Hi, "HI");
                    ui.selectable_value(&mut self.language, LanguageChoice::Zh, "ZH");
                    ui.selectable_value(&mut self.language, LanguageChoice::ZhTw, "ZH-TW");
                    ui.selectable_value(&mut self.language, LanguageChoice::Ko, "KO");
                    ui.selectable_value(&mut self.language, LanguageChoice::Ja, "JA");
                    ui.selectable_value(&mut self.language, LanguageChoice::Ur, "UR");
                    ui.selectable_value(&mut self.language, LanguageChoice::Fa, "FA");
                    ui.selectable_value(&mut self.language, LanguageChoice::Tr, "TR");
                    ui.selectable_value(&mut self.language, LanguageChoice::Az, "AZ");
                    ui.selectable_value(&mut self.language, LanguageChoice::Hy, "HY");
                });
                ui.label(format!("selected: {}", self.language.label()));
                if self.language == LanguageChoice::ZhTw {
                    ui.small("Whisper uses zh for ZH-TW");
                }
                if self.language == LanguageChoice::Fa {
                    ui.small("Iranian/Persian → fa");
                }

                ui.add_space(8.0);
                ui.label("Whisper model");
                ui.vertical(|ui| {
                    for size in WhisperSize::ALL {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.whisper_size, size, size.label());
                        });
                    }
                });
                let present = self.whisper_size.exists();
                if present {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 180, 80),
                        format!("✓ present — {}", self.whisper_size.filename()),
                    );
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 140, 40),
                        format!(
                            "✗ missing — run scripts/download-models.sh ({})",
                            self.whisper_size.filename()
                        ),
                    );
                }

                ui.add_space(8.0);
                ui.checkbox(&mut self.tts_on, "TTS on");
                ui.checkbox(&mut self.enable_vad, "VAD");
                ui.checkbox(&mut self.enable_mt, "Translate → EN");

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.running, egui::Button::new("Start"))
                        .clicked()
                    {
                        self.start();
                    }
                    if ui
                        .add_enabled(self.running, egui::Button::new("Stop"))
                        .clicked()
                    {
                        self.stop();
                    }
                    if ui.button("Clear").clicked() {
                        self.transcript.clear();
                        self.translation.clear();
                        self.placeholders = false;
                        self.error.clear();
                    }
                });

                ui.add_space(16.0);
                ui.heading("EP status");
                for line in &self.ep_lines {
                    ui.monospace(line);
                }

                ui.add_space(12.0);
                ui.heading("VAD");
                ui.monospace(&self.last_vad);

                if !self.error.is_empty() {
                    ui.add_space(12.0);
                    ui.colored_label(egui::Color32::RED, "Error");
                    ui.colored_label(egui::Color32::LIGHT_RED, &self.error);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Live transcript");
            ui.columns(2, |cols| {
                cols[0].heading("Original");
                egui::ScrollArea::vertical()
                    .id_salt("original_scroll")
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(&mut cols[0], |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.transcript)
                                .desired_width(f32::INFINITY)
                                .desired_rows(24)
                                .font(egui::TextStyle::Body),
                        );
                    });

                cols[1].heading("English");
                egui::ScrollArea::vertical()
                    .id_salt("english_scroll")
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(&mut cols[1], |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.translation)
                                .desired_width(f32::INFINITY)
                                .desired_rows(24)
                                .font(egui::TextStyle::Body),
                        );
                    });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Prefer cwd = crate root so profiles/models resolve.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let _ = std::env::set_current_dir(&manifest_dir);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("edge-ort-gui"),
        ..Default::default()
    };
    eframe::run_native(
        "edge-ort-gui",
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(GuiApp::default()))
        }),
    )
}
