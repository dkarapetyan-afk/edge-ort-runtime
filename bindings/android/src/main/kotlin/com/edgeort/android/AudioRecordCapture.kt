package com.edgeort.android

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import uniffi.edge_ort_runtime.NativePipelineHandle
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Audio capture modes supported by Edge ORT.
 */
enum class AudioCaptureSource(val label: String) {
    ALL_AUDIO("All Audio (Speakers + Mic)"),
    SPEAKERS_ONLY("Speakers / Playback Only"),
    MIC_ONLY("Microphone Only");

    companion object {
        fun fromName(name: String?): AudioCaptureSource {
            return entries.firstOrNull { it.name.equals(name, ignoreCase = true) } ?: ALL_AUDIO
        }
    }
}

/**
 * Multi-source Android audio capture with real-time digital mixing.
 *
 * Supports:
 * 1. Speakers / Device Playback audio via [AudioPlaybackCaptureConfiguration] (Android 10+ / API 29+)
 * 2. Pure ambient microphone audio via [MediaRecorder.AudioSource.MIC]
 * 3. Concurrent dual-stream capture with soft-clipping audio mixing for "All Audio"
 *
 * Pushes 16 kHz Mono PCM16 chunks directly to [NativePipelineHandle].
 */
class AudioRecordCapture(
    private val pipeline: NativePipelineHandle,
    private val sourceMode: AudioCaptureSource = AudioCaptureSource.ALL_AUDIO,
    private val mediaProjection: MediaProjection? = null,
    private val sampleRate: Int = 16000
) {
    companion object {
        private const val TAG = "AudioRecordCapture"
        private const val CHUNK_DURATION_MS = 50 // 50ms buffer chunks
    }

    private val chunkSize = (sampleRate * CHUNK_DURATION_MS) / 1000 // 800 samples at 16 kHz
    private val isRecording = AtomicBoolean(false)

    private var micRecord: AudioRecord? = null
    private var playbackRecord: AudioRecord? = null

    private var micThread: Thread? = null
    private var playbackThread: Thread? = null
    private var mixerThread: Thread? = null

    private val micQueue = ConcurrentLinkedQueue<ShortArray>()
    private val playbackQueue = ConcurrentLinkedQueue<ShortArray>()

    private val projectionCallback = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
        object : MediaProjection.Callback() {
            override fun onStop() {
                Log.i(TAG, "MediaProjection stopped by system/user")
            }
        }
    } else null

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (isRecording.get()) return true

        val channelConfigMono = AudioFormat.CHANNEL_IN_MONO
        val encodingPcm16 = AudioFormat.ENCODING_PCM_16BIT
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfigMono, encodingPcm16)
        val bufferSize = (minBufferSize * 2).coerceAtLeast(sampleRate / 10)

        val needMic = sourceMode == AudioCaptureSource.ALL_AUDIO || sourceMode == AudioCaptureSource.MIC_ONLY
        val needPlayback = (sourceMode == AudioCaptureSource.ALL_AUDIO || sourceMode == AudioCaptureSource.SPEAKERS_ONLY) &&
                mediaProjection != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q

        // 1. Initialize Microphone Recorder if needed
        if (needMic) {
            try {
                val mic = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfigMono,
                    encodingPcm16,
                    bufferSize
                )
                if (mic.state == AudioRecord.STATE_INITIALIZED) {
                    micRecord = mic
                    Log.i(TAG, "Microphone AudioRecord initialized successfully")
                } else {
                    Log.w(TAG, "Microphone AudioRecord failed to initialize")
                    mic.release()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Failed creating microphone AudioRecord: ${t.message}", t)
            }
        }

        // 2. Initialize Playback (Speakers) Recorder if requested and projection is available
        if (needPlayback && mediaProjection != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                projectionCallback?.let {
                    mediaProjection.registerCallback(it, Handler(Looper.getMainLooper()))
                }

                val captureConfig = AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
                    .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                    .addMatchingUsage(AudioAttributes.USAGE_GAME)
                    .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
                    .build()

                val audioFormat = AudioFormat.Builder()
                    .setEncoding(encodingPcm16)
                    .setSampleRate(sampleRate)
                    .setChannelMask(channelConfigMono)
                    .build()

                var pbRecord = AudioRecord.Builder()
                    .setAudioPlaybackCaptureConfig(captureConfig)
                    .setAudioFormat(audioFormat)
                    .setBufferSizeInBytes(bufferSize)
                    .build()

                if (pbRecord.state != AudioRecord.STATE_INITIALIZED) {
                    pbRecord.release()
                    // Fallback to stereo if mono is unsupported by device HAL
                    val stereoFormat = AudioFormat.Builder()
                        .setEncoding(encodingPcm16)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                        .build()
                    pbRecord = AudioRecord.Builder()
                        .setAudioPlaybackCaptureConfig(captureConfig)
                        .setAudioFormat(stereoFormat)
                        .setBufferSizeInBytes(bufferSize * 2)
                        .build()
                }

                if (pbRecord.state == AudioRecord.STATE_INITIALIZED) {
                    playbackRecord = pbRecord
                    Log.i(TAG, "Playback (Speakers) AudioRecord initialized successfully")
                } else {
                    Log.w(TAG, "Playback AudioRecord failed initialization on this device")
                    pbRecord.release()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Failed creating Playback AudioRecord: ${t.message}", t)
            }
        }

        // If neither initialized, fail
        if (micRecord == null && playbackRecord == null) {
            Log.e(TAG, "No audio capture sources could be initialized")
            return false
        }

        isRecording.set(true)

        // Start native recording hardware
        try {
            micRecord?.startRecording()
        } catch (t: Throwable) {
            Log.e(TAG, "Failed starting mic recording: ${t.message}", t)
        }

        try {
            playbackRecord?.startRecording()
        } catch (t: Throwable) {
            Log.e(TAG, "Failed starting playback recording: ${t.message}", t)
        }

        // If both sources are running, launch reader threads + mixer thread
        if (micRecord != null && playbackRecord != null) {
            startDualCaptureAndMixer()
        } else if (playbackRecord != null) {
            startSinglePlaybackCapture()
        } else {
            startSingleMicCapture()
        }

        return true
    }

    private fun startDualCaptureAndMixer() {
        val mic = micRecord ?: return
        val play = playbackRecord ?: return

        // 1. Mic Reader Thread
        micThread = Thread({
            val buf = ShortArray(chunkSize)
            while (isRecording.get()) {
                val read = mic.read(buf, 0, buf.size)
                if (read > 0) {
                    val copy = if (read == buf.size) buf.clone() else buf.copyOf(read)
                    micQueue.offer(copy)
                    // Keep queue bounded to avoid latency buildup
                    while (micQueue.size > 10) micQueue.poll()
                }
            }
        }, "EdgeOrtMicCaptureThread").apply { start() }

        // 2. Playback (Speakers) Reader Thread
        playbackThread = Thread({
            val isStereo = play.channelCount == 2
            val readBufSize = if (isStereo) chunkSize * 2 else chunkSize
            val buf = ShortArray(readBufSize)

            while (isRecording.get()) {
                val read = play.read(buf, 0, buf.size)
                if (read > 0) {
                    val mono = if (isStereo) {
                        val monoLen = read / 2
                        ShortArray(monoLen) { i ->
                            val l = buf[i * 2].toInt()
                            val r = buf[i * 2 + 1].toInt()
                            ((l + r) / 2).toShort()
                        }
                    } else {
                        if (read == buf.size) buf.clone() else buf.copyOf(read)
                    }
                    playbackQueue.offer(mono)
                    // Keep queue bounded
                    while (playbackQueue.size > 10) playbackQueue.poll()
                }
            }
        }, "EdgeOrtPlaybackCaptureThread").apply { start() }

        // 3. Audio Mixer Thread
        mixerThread = Thread({
            val mixed = ShortArray(chunkSize)

            while (isRecording.get()) {
                val micChunk = micQueue.poll()
                val playChunk = playbackQueue.poll()

                if (micChunk == null && playChunk == null) {
                    try {
                        Thread.sleep(10)
                    } catch (_: InterruptedException) {
                        break
                    }
                    continue
                }

                val maxLen = maxOf(micChunk?.size ?: 0, playChunk?.size ?: 0).coerceAtMost(chunkSize)
                if (maxLen == 0) continue

                for (i in 0 until maxLen) {
                    val sMic = if (micChunk != null && i < micChunk.size) micChunk[i].toInt() else 0
                    val sPlay = if (playChunk != null && i < playChunk.size) playChunk[i].toInt() else 0
                    // Saturated 16-bit PCM addition
                    mixed[i] = (sMic + sPlay).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
                }

                val slice = if (maxLen == chunkSize) mixed.toList() else mixed.take(maxLen)
                pipeline.pushPcm16(slice)
            }
        }, "EdgeOrtAudioMixerThread").apply { start() }
    }

    private fun startSingleMicCapture() {
        val mic = micRecord ?: return
        micThread = Thread({
            val audioBuffer = ShortArray(chunkSize)
            while (isRecording.get()) {
                val readCount = mic.read(audioBuffer, 0, audioBuffer.size)
                if (readCount > 0) {
                    val slice = if (readCount == audioBuffer.size) {
                        audioBuffer.toList()
                    } else {
                        audioBuffer.take(readCount)
                    }
                    pipeline.pushPcm16(slice)
                }
            }
        }, "EdgeOrtMicCaptureThread").apply { start() }
    }

    private fun startSinglePlaybackCapture() {
        val play = playbackRecord ?: return
        playbackThread = Thread({
            val isStereo = play.channelCount == 2
            val readBufSize = if (isStereo) chunkSize * 2 else chunkSize
            val audioBuffer = ShortArray(readBufSize)

            while (isRecording.get()) {
                val readCount = play.read(audioBuffer, 0, audioBuffer.size)
                if (readCount > 0) {
                    if (isStereo) {
                        val monoLen = readCount / 2
                        val mono = ShortArray(monoLen) { i ->
                            val l = audioBuffer[i * 2].toInt()
                            val r = audioBuffer[i * 2 + 1].toInt()
                            ((l + r) / 2).toShort()
                        }
                        pipeline.pushPcm16(mono.toList())
                    } else {
                        val slice = if (readCount == audioBuffer.size) {
                            audioBuffer.toList()
                        } else {
                            audioBuffer.take(readCount)
                        }
                        pipeline.pushPcm16(slice)
                    }
                }
            }
        }, "EdgeOrtPlaybackCaptureThread").apply { start() }
    }

    fun stop() {
        if (!isRecording.getAndSet(false)) return

        // Interrupt threads
        micThread?.interrupt()
        playbackThread?.interrupt()
        mixerThread?.interrupt()

        // Stop AudioRecords
        micRecord?.apply {
            try {
                stop()
                release()
            } catch (_: Exception) {}
        }
        micRecord = null

        playbackRecord?.apply {
            try {
                stop()
                release()
            } catch (_: Exception) {}
        }
        playbackRecord = null

        // Unregister projection callback
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && projectionCallback != null) {
            try {
                mediaProjection?.unregisterCallback(projectionCallback)
            } catch (_: Exception) {}
        }

        try {
            micThread?.join(300)
            playbackThread?.join(300)
            mixerThread?.join(300)
        } catch (_: Exception) {}

        micThread = null
        playbackThread = null
        mixerThread = null

        micQueue.clear()
        playbackQueue.clear()
        Log.i(TAG, "AudioRecordCapture stopped and cleaned up")
    }

    fun isRunning(): Boolean = isRecording.get()
}
