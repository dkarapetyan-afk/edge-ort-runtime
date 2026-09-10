//! PipeWire mic / sink-monitor capture → PCM bus (f32 mono @ 16 kHz).

use super::{
    downmix_to_mono, resample_linear, AudioError, AudioSource, PCM_SAMPLE_RATE,
};
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Capture target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Default input (microphone).
    Mic,
    /// Default sink monitor (system mix).
    Monitor,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mic => "mic",
            Self::Monitor => "monitor",
        }
    }
}

struct CaptureFormat {
    rate: u32,
    channels: u32,
}

enum PwMsg {
    Samples(Vec<f32>),
    Format(CaptureFormat),
    Error(String),
    Done,
}

/// Background PipeWire capture thread feeding a queue of mono 16 kHz samples.
pub struct PipeWireSource {
    kind: SourceKind,
    rx: Receiver<PwMsg>,
    queue: VecDeque<f32>,
    join: Option<JoinHandle<()>>,
    stop_tx: Sender<()>,
    closed: bool,
}

impl PipeWireSource {
    pub fn open(kind: SourceKind) -> Result<Self, AudioError> {
        let (tx, rx) = mpsc::channel::<PwMsg>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let join = thread::Builder::new()
            .name(format!("pw-{}", kind.as_str()))
            .spawn(move || {
                if let Err(e) = run_capture(kind, tx.clone(), stop_rx) {
                    let _ = tx.send(PwMsg::Error(e));
                }
                let _ = tx.send(PwMsg::Done);
            })
            .map_err(|e| AudioError::PipeWire(format!("spawn: {e}")))?;

        Ok(Self {
            kind,
            rx,
            queue: VecDeque::new(),
            join: Some(join),
            stop_tx,
            closed: false,
        })
    }
}

impl AudioSource for PipeWireSource {
    fn name(&self) -> &str {
        self.kind.as_str()
    }

    fn pull(&mut self) -> Result<Vec<f32>, AudioError> {
        if self.closed && self.queue.is_empty() {
            return Ok(Vec::new());
        }

        // Drain messages briefly.
        let deadline = std::time::Instant::now() + Duration::from_millis(120);
        loop {
            match self.rx.try_recv() {
                Ok(PwMsg::Samples(s)) => self.queue.extend(s),
                Ok(PwMsg::Format(f)) => {
                    debug!(rate = f.rate, channels = f.channels, "pipewire format");
                }
                Ok(PwMsg::Error(e)) => return Err(AudioError::PipeWire(e)),
                Ok(PwMsg::Done) => {
                    self.closed = true;
                }
                Err(TryRecvError::Empty) => {
                    if !self.queue.is_empty() || self.closed {
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
            if self.queue.len() >= (PCM_SAMPLE_RATE as usize / 10) {
                break;
            }
        }

        let want = PCM_SAMPLE_RATE as usize / 10; // 100 ms
        if self.queue.is_empty() {
            if self.closed {
                return Ok(Vec::new());
            }
            // No data yet — return silence chunk so pipeline keeps ticking.
            return Ok(vec![0.0; want]);
        }

        let n = want.min(self.queue.len());
        Ok(self.queue.drain(..n).collect())
    }
}

impl Drop for PipeWireSource {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

fn run_capture(kind: SourceKind, tx: Sender<PwMsg>, stop_rx: Receiver<()>) -> Result<(), String> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| e.to_string())?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(|e| e.to_string())?;
    let core = context.connect_rc(None).map_err(|e| {
        format!(
            "failed to connect to PipeWire (is the daemon running?): {e}"
        )
    })?;

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Speech",
    };
    if kind == SourceKind::Monitor {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }

    info!(source = kind.as_str(), "opening PipeWire capture stream");

    let stream = pw::stream::StreamBox::new(&core, "edge-ort-runtime", props)
        .map_err(|e| e.to_string())?;

    struct UserData {
        format: spa::param::audio::AudioInfoRaw,
        tx: Sender<PwMsg>,
    }

    let data = UserData {
        format: Default::default(),
        tx: tx.clone(),
    };

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .param_changed(|_, user_data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            if user_data.format.parse(param).is_ok() {
                let _ = user_data.tx.send(PwMsg::Format(CaptureFormat {
                    rate: user_data.format.rate(),
                    channels: user_data.format.channels(),
                }));
            }
        })
        .process(|stream, user_data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let channels = user_data.format.channels().max(1) as usize;
            let rate = user_data.format.rate();
            if rate == 0 {
                return;
            }
            let n_bytes = data.chunk().size() as usize;
            if n_bytes == 0 {
                return;
            }
            let Some(bytes) = data.data() else {
                return;
            };
            let n_samples = n_bytes / std::mem::size_of::<f32>();
            let mut interleaved = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let start = i * 4;
                if start + 4 > bytes.len() {
                    break;
                }
                let s = f32::from_le_bytes([
                    bytes[start],
                    bytes[start + 1],
                    bytes[start + 2],
                    bytes[start + 3],
                ]);
                interleaved.push(s);
            }
            let mono = downmix_to_mono(&interleaved, channels);
            let bus = resample_linear(&mono, rate, PCM_SAMPLE_RATE);
            if !bus.is_empty() {
                let _ = user_data.tx.send(PwMsg::Samples(bus));
            }
        })
        .register()
        .map_err(|e| e.to_string())?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| format!("pod serialize: {e}"))?
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).ok_or("invalid format pod")?];

    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| e.to_string())?;

    // Drive the loop until stop signal.
    use pw::loop_::Timeout;
    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => {
                warn!("stopping PipeWire capture");
                mainloop.quit();
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        mainloop
            .loop_()
            .iterate(Timeout::Finite(Duration::from_millis(50)));
    }

    Ok(())
}
