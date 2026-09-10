//! PipeWire PCM playback (f32 mono → default sink). Falls back to logging on failure.

use super::PCM_SAMPLE_RATE;
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::pod::Pod;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::{info, warn};

struct PlaybackState {
    queue: VecDeque<f32>,
    done: bool,
    error: Option<String>,
}

/// Play mono f32 PCM via PipeWire. Returns Ok when playback finishes or is stubbed.
pub fn play_pcm(pcm: &[f32], sample_rate: u32) -> Result<(), String> {
    if pcm.is_empty() {
        return Ok(());
    }

    let rate = if sample_rate == 0 {
        PCM_SAMPLE_RATE
    } else {
        sample_rate
    };

    match play_pcm_pipewire(pcm, rate) {
        Ok(()) => Ok(()),
        Err(e) => {
            warn!(error = %e, "PipeWire playback failed — logging stub");
            playback_log(pcm, rate);
            Ok(())
        }
    }
}

/// Always-available logging path (also used when PW is down).
pub fn playback_log(pcm: &[f32], sample_rate: u32) {
    if pcm.is_empty() {
        return;
    }
    let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
    info!(
        samples = pcm.len(),
        sample_rate,
        rms,
        "tts playback (log stub)"
    );
}

fn play_pcm_pipewire(pcm: &[f32], sample_rate: u32) -> Result<(), String> {
    let state = Arc::new(Mutex::new(PlaybackState {
        queue: pcm.iter().copied().collect(),
        done: false,
        error: None,
    }));
    let state_thread = Arc::clone(&state);

    let join: JoinHandle<Result<(), String>> = thread::Builder::new()
        .name("pw-playback".into())
        .spawn(move || run_playback(state_thread, sample_rate))
        .map_err(|e| format!("spawn playback: {e}"))?;

    // Wait until queue drains or error (with timeout based on audio length + slack).
    let secs = pcm.len() as f64 / sample_rate as f64 + 2.0;
    let deadline = std::time::Instant::now() + Duration::from_secs_f64(secs.max(1.0));
    loop {
        {
            let st = state.lock().map_err(|e| e.to_string())?;
            if let Some(ref e) = st.error {
                let _ = join.join();
                return Err(e.clone());
            }
            if st.done || st.queue.is_empty() {
                break;
            }
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    // Signal done and join briefly.
    {
        if let Ok(mut st) = state.lock() {
            st.done = true;
        }
    }
    // Give the PW thread a moment to exit; don't block forever.
    let _ = join.join();
    Ok(())
}

fn run_playback(state: Arc<Mutex<PlaybackState>>, sample_rate: u32) -> Result<(), String> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| e.to_string())?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(|e| e.to_string())?;
    let core = context.connect_rc(None).map_err(|e| {
        format!("failed to connect to PipeWire for playback (is the daemon running?): {e}")
    })?;

    let stream = pw::stream::StreamBox::new(
        &core,
        "edge-ort-tts",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Speech",
            *pw::keys::AUDIO_CHANNELS => "1",
        },
    )
    .map_err(|e| e.to_string())?;

    struct UserData {
        state: Arc<Mutex<PlaybackState>>,
    }

    let data = UserData {
        state: Arc::clone(&state),
    };

    let _listener = stream
        .add_local_listener_with_user_data(data)
        .process(|stream, user_data| {
            let Ok(mut st) = user_data.state.lock() else {
                return;
            };
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let Some(slice) = data.data() else {
                return;
            };
            let n_bytes = slice.len();
            let n_samples = n_bytes / std::mem::size_of::<f32>();
            for i in 0..n_samples {
                let sample = st.queue.pop_front().unwrap_or(0.0);
                let bytes = sample.to_le_bytes();
                let start = i * 4;
                if start + 4 <= slice.len() {
                    slice[start..start + 4].copy_from_slice(&bytes);
                }
            }
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = 4;
            *chunk.size_mut() = (n_samples * 4) as _;

            if st.queue.is_empty() {
                st.done = true;
                // Quit mainloop from process is awkward; flag is enough — outer wait exits.
            }
        })
        .register()
        .map_err(|e| e.to_string())?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(sample_rate);
    audio_info.set_channels(1);

    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: pw::spa::param::ParamType::EnumFormat.as_raw(),
            properties: audio_info.into(),
        }),
    )
    .map_err(|e| format!("pod serialize: {e}"))?
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).ok_or("invalid format pod")?];

    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| e.to_string())?;

    use pw::loop_::Timeout;
    let start = std::time::Instant::now();
    let max = Duration::from_secs(30);
    loop {
        {
            let st = state.lock().map_err(|e| e.to_string())?;
            if st.done && st.queue.is_empty() {
                mainloop.quit();
                break;
            }
            if let Some(ref e) = st.error {
                return Err(e.clone());
            }
        }
        if start.elapsed() > max {
            mainloop.quit();
            break;
        }
        mainloop
            .loop_()
            .iterate(Timeout::Finite(Duration::from_millis(20)));
    }

    Ok(())
}
