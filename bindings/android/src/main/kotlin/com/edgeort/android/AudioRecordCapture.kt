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
    private val sampleRate: Int = 16000,
    private val disableMic: Boolean = false,
    var onAudioPcm: ((ByteArray) -> Unit)? = null
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

        val needMic = !disableMic && (sourceMode == AudioCaptureSource.ALL_AUDIO || sourceMode == AudioCaptureSource.MIC_ONLY)
        val needPlayback = (sourceMode == AudioCaptureSource.ALL_AUDIO || sourceMode == AudioCaptureSource.SPEAKERS_ONLY) &&
                mediaProjection != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q

        // 1. Initialize Microphone Recorder if needed (use VOICE_RECOGNITION for cooperative concurrent capture with Google SODA / TPU)
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

        // 1. Playback (Speakers) Reader Thread
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
                    // Keep queue bounded to avoid latency
                    while (playbackQueue.size > 5) playbackQueue.poll()
                }
            }
        }, "EdgeOrtPlaybackCaptureThread").apply { start() }

        // 2. Mic Reader & Digital Mixer Thread (Master Clock: 16 kHz, 50ms chunks)
        micThread = Thread({
            val micBuf = ShortArray(chunkSize)
            val mixed = ShortArray(chunkSize)
            var lastLogTime = System.currentTimeMillis()
            var chunkCount = 0L

            while (isRecording.get()) {
                val micRead = mic.read(micBuf, 0, micBuf.size)
                if (micRead <= 0) continue

                chunkCount++
                val playChunk = playbackQueue.poll()

                var maxMicAmp = 0
                var maxPlayAmp = 0

                for (i in 0 until micRead) {
                    val sMic = micBuf[i].toInt()
                    val sPlay = if (playChunk != null && i < playChunk.size) playChunk[i].toInt() else 0
                    val absMic = kotlin.math.abs(sMic)
                    val absPlay = kotlin.math.abs(sPlay)
                    if (absMic > maxMicAmp) maxMicAmp = absMic
                    if (absPlay > maxPlayAmp) maxPlayAmp = absPlay

                    // Saturated 16-bit PCM addition
                    mixed[i] = (sMic + sPlay).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
                }

                val slice = if (micRead == chunkSize) mixed.toList() else mixed.take(micRead)
                pipeline.pushPcm16(slice)
                onAudioPcm?.let { cb ->
                    val byteArr = ByteArray(slice.size * 2)
                    for (idx in slice.indices) {
                        val s = slice[idx]
                        byteArr[idx * 2] = (s.toInt() and 0xFF).toByte()
                        byteArr[idx * 2 + 1] = ((s.toInt() shr 8) and 0xFF).toByte()
                    }
                    cb(byteArr)
                }

                val now = System.currentTimeMillis()
                if (now - lastLogTime >= 3000) {
                    Log.d(TAG, "AudioStream: chunks=$chunkCount micMaxAmp=$maxMicAmp playMaxAmp=$maxPlayAmp playQueued=${playbackQueue.size}")
                    lastLogTime = now
                }
            }
        }, "EdgeOrtMicAndMixerThread").apply { start() }
    }

    private fun startSingleMicCapture() {
        val mic = micRecord ?: return
        micThread = Thread({
            val audioBuffer = ShortArray(chunkSize)
            var lastLogTime = System.currentTimeMillis()
            var chunkCount = 0L

            while (isRecording.get()) {
                val readCount = mic.read(audioBuffer, 0, audioBuffer.size)
                if (readCount > 0) {
                    chunkCount++
                    val slice = if (readCount == audioBuffer.size) {
                        audioBuffer.toList()
                    } else {
                        audioBuffer.take(readCount)
                    }
                    pipeline.pushPcm16(slice)

                    val now = System.currentTimeMillis()
                    if (now - lastLogTime >= 3000) {
                        var maxAmp = 0
                        for (i in 0 until readCount) {
                            val a = kotlin.math.abs(audioBuffer[i].toInt())
                            if (a > maxAmp) maxAmp = a
                        }
                        Log.d(TAG, "MicOnly: chunks=$chunkCount maxAmp=$maxAmp")
                        lastLogTime = now
                    }
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
            var lastLogTime = System.currentTimeMillis()
            var chunkCount = 0L

            while (isRecording.get()) {
                val readCount = play.read(audioBuffer, 0, audioBuffer.size)
                if (readCount > 0) {
                    chunkCount++
                    val monoList: List<Short>
                    val monoBytes: ByteArray
                    if (isStereo) {
                        val monoLen = readCount / 2
                        val mono = ShortArray(monoLen) { i ->
                            val l = audioBuffer[i * 2].toInt()
                            val r = audioBuffer[i * 2 + 1].toInt()
                            ((l + r) / 2).toShort()
                        }
                        monoList = mono.toList()
                        monoBytes = ByteArray(monoLen * 2)
                        for (idx in 0 until monoLen) {
                            val s = mono[idx]
                            monoBytes[idx * 2] = (s.toInt() and 0xFF).toByte()
                            monoBytes[idx * 2 + 1] = ((s.toInt() shr 8) and 0xFF).toByte()
                        }
                    } else {
                        val mono = if (readCount == audioBuffer.size) audioBuffer else audioBuffer.copyOf(readCount)
                        monoList = mono.toList()
                        monoBytes = ByteArray(mono.size * 2)
                        for (idx in mono.indices) {
                            val s = mono[idx]
                            monoBytes[idx * 2] = (s.toInt() and 0xFF).toByte()
                            monoBytes[idx * 2 + 1] = ((s.toInt() shr 8) and 0xFF).toByte()
                        }
                    }
                    pipeline.pushPcm16(monoList)
                    onAudioPcm?.invoke(monoBytes)

                    val now = System.currentTimeMillis()
                    if (now - lastLogTime >= 3000) {
                        Log.d(TAG, "PlaybackOnly: chunks=$chunkCount")
                        lastLogTime = now
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
