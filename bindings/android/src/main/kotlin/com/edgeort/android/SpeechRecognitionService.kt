package com.edgeort.android

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import uniffi.edge_ort_runtime.*
import android.media.AudioFormat
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.OutputStream
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Android Foreground Service for persistent, background speech recognition and translation.
 * Captures all audio including device speakers (media playback) and ambient microphone,
 * routing mixed PCM chunks through the native Edge ORT pipeline.
 */
class SpeechRecognitionService : Service() {

    companion object {
        const val CHANNEL_ID = "edge_ort_speech_channel"
        const val NOTIFICATION_ID = 1001

        const val ACTION_START = "com.edgeort.action.START"
        const val ACTION_STOP = "com.edgeort.action.STOP"
        const val ACTION_UPDATE_LANGUAGE = "com.edgeort.action.UPDATE_LANGUAGE"
        const val ACTION_TEST_SPEECH = "com.edgeort.action.TEST_SPEECH"

        const val EXTRA_LANGUAGE = "extra_language"
        const val EXTRA_AUDIO_SOURCE = "extra_audio_source"
        const val EXTRA_PROJECTION_RESULT_CODE = "extra_projection_result_code"
        const val EXTRA_PROJECTION_DATA = "extra_projection_data"
        const val EXTRA_TEST_TEXT = "extra_test_text"
        const val EXTRA_TEST_LANG = "extra_test_lang"

        private const val TAG = "SpeechRecService"

        // Event flow for activities and UI overlays
        private val _events = MutableSharedFlow<PipelineUpdate>(extraBufferCapacity = 64)
        val events: SharedFlow<PipelineUpdate> = _events.asSharedFlow()

        private val _isRunning = MutableStateFlow(false)
        val isRunning: StateFlow<Boolean> = _isRunning.asStateFlow()

        init {
            try {
                System.loadLibrary("c++_shared")
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to load c++_shared: ${t.message}")
            }
        }
    }

    sealed class PipelineUpdate {
        data class Status(val message: String) : PipelineUpdate()
        data class Vad(val speechProb: Float, val isSpeech: Boolean) : PipelineUpdate()
        data class Transcript(val text: String, val translation: String?, val confidence: Float) : PipelineUpdate()
        data class Error(val message: String) : PipelineUpdate()
        object Done : PipelineUpdate()
    }

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var wakeLock: PowerManager.WakeLock? = null
    private var pipeline: NativePipelineHandle? = null
    private var audioCapture: AudioRecordCapture? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var speechRecognizer: SpeechRecognizer? = null
    private val isListeningActive = AtomicBoolean(false)
    private val hasTpuTranscript = AtomicBoolean(false)
    private val translationService by lazy { DeviceTranslationService(this) }
    private var currentLanguage: String = "en"

    private class AudioPipeManager {
        @Volatile private var currentOutputStream: OutputStream? = null
        @Volatile private var currentWritePfd: ParcelFileDescriptor? = null
        @Volatile private var currentReadPfd: ParcelFileDescriptor? = null

        @Synchronized
        fun createPipe(): ParcelFileDescriptor? {
            close()
            return try {
                val pipe = ParcelFileDescriptor.createPipe()
                currentReadPfd = pipe[0]
                currentWritePfd = pipe[1]
                currentOutputStream = ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])
                pipe[0]
            } catch (t: Throwable) {
                Log.e(TAG, "Failed creating audio pipe: ${t.message}")
                null
            }
        }

        fun write(bytes: ByteArray) {
            val stream = currentOutputStream ?: return
            try {
                stream.write(bytes)
                stream.flush()
            } catch (_: Throwable) {
                close()
            }
        }

        @Synchronized
        fun close() {
            try { currentOutputStream?.close() } catch (_: Throwable) {}
            try { currentWritePfd?.close() } catch (_: Throwable) {}
            try { currentReadPfd?.close() } catch (_: Throwable) {}
            currentOutputStream = null
            currentWritePfd = null
            currentReadPfd = null
        }
    }

    private val audioPipeManager = AudioPipeManager()

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "EdgeOrt::SpeechRecognitionWakeLock"
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand: action=${intent?.action}")
        when (intent?.action) {
            ACTION_STOP -> {
                Log.i(TAG, "stopService requested via ACTION_STOP")
                stopService()
                return START_NOT_STICKY
            }
            ACTION_UPDATE_LANGUAGE -> {
                val newLang = intent.getStringExtra(EXTRA_LANGUAGE) ?: "en"
                Log.i(TAG, "Language update requested: $newLang")
                currentLanguage = newLang
                if (isListeningActive.get()) {
                    startTpuSpeechRecognizer(newLang)
                }
                return START_STICKY
            }
            ACTION_TEST_SPEECH -> {
                val notification = buildNotification("Edge ORT: Testing Speech & Translation")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
                } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
                } else {
                    startForeground(NOTIFICATION_ID, notification)
                }
                val text = intent.getStringExtra(EXTRA_TEST_TEXT) ?: "Hello world, testing speech recognition and translation"
                val lang = intent.getStringExtra(EXTRA_TEST_LANG) ?: "en"
                Log.i(TAG, "ACTION_TEST_SPEECH received: text='$text', lang='$lang'")
                serviceScope.launch {
                    val trans = translationService.translateToEnglish(text, lang)
                    _events.emit(PipelineUpdate.Transcript(text, trans, 0.99f))
                    val displayText = if (!trans.isNullOrBlank() && !trans.startsWith("Original")) "$text → $trans" else text
                    updateNotification(displayText)
                }
                return START_STICKY
            }
            ACTION_START -> {
                val lang = intent.getStringExtra(EXTRA_LANGUAGE) ?: "auto"
                val sourceModeStr = intent.getStringExtra(EXTRA_AUDIO_SOURCE)
                val sourceMode = AudioCaptureSource.fromName(sourceModeStr)
                val projCode = intent.getIntExtra(EXTRA_PROJECTION_RESULT_CODE, Activity.RESULT_CANCELED)
                val projData: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_PROJECTION_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_PROJECTION_DATA)
                }

                startPipeline(lang, sourceMode, projCode, projData)
            }
        }
        return START_STICKY
    }

    private fun startPipeline(
        language: String,
        sourceMode: AudioCaptureSource,
        projectionResultCode: Int,
        projectionData: Intent?
    ) {
        val hasProjection = projectionResultCode == Activity.RESULT_OK && projectionData != null
        val sourceLabel = when (sourceMode) {
            AudioCaptureSource.ALL_AUDIO -> if (hasProjection) "All Audio (Speakers + Mic)" else "Ambient Audio (Mic & Room)"
            AudioCaptureSource.SPEAKERS_ONLY -> "Device Speakers / Playback"
            AudioCaptureSource.MIC_ONLY -> "Microphone Only"
        }

        val notification = buildNotification("Listening: $sourceLabel")

        // Start Foreground Service with required types for Android 14+ (API 34/35/36)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            val fgsType = if (hasProjection) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            } else {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            }
            startForeground(NOTIFICATION_ID, notification, fgsType)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val fgsType = if (hasProjection) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            } else {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            }
            startForeground(NOTIFICATION_ID, notification, fgsType)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        wakeLock?.acquire(3 * 60 * 60 * 1000L) // Max 3 hours safety timeout

        // Acquire MediaProjection after FGS is started (mandatory on Android 14+)
        var mediaProjection: MediaProjection? = null
        if (hasProjection && projectionData != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            try {
                val mpManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
                mediaProjection = mpManager.getMediaProjection(projectionResultCode, projectionData)
                Log.i(TAG, "MediaProjection successfully acquired for internal speaker playback capture")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed obtaining MediaProjection: ${t.message}", t)
            }
        }

        isListeningActive.set(true)
        hasTpuTranscript.set(false)
        _isRunning.value = true

        try {
            val profilePath = File(filesDir, "profiles/default").takeIf { it.exists() }?.absolutePath
            val handle = NativePipelineHandle(
                languageCode = language,
                enableVad = true,
                enableMt = true,
                profilePath = profilePath
            )
            pipeline = handle

            handle.start(object : NativeSpeechListener {
                override fun onStatus(status: String) {
                    Log.i(TAG, "onStatus: $status")
                    serviceScope.launch { _events.emit(PipelineUpdate.Status(status)) }
                    updateNotification(status)
                }

                private var lastVadLogTime = 0L
                override fun onVad(speechProb: Float, isSpeech: Boolean) {
                    val now = System.currentTimeMillis()
                    if (isSpeech && now - lastVadLogTime > 5000L) {
                        Log.d(TAG, "onVad: speechProb=$speechProb, isSpeech=$isSpeech")
                        lastVadLogTime = now
                    }
                    serviceScope.launch { _events.emit(PipelineUpdate.Vad(speechProb, isSpeech)) }
                }

                override fun onTranscript(text: String, translation: String?, confidence: Float) {
                    if (text.startsWith("[stub-asr") || text.startsWith("[Speech captured") || text.startsWith("[Silence")) {
                        // Suppress diagnostic stubs from polluting the UI transcript card
                        return
                    }
                    if (hasTpuTranscript.get()) {
                        // Tensor TPU provides higher-quality live ASR
                        return
                    }
                    Log.i(TAG, "onTranscript: '$text' (translation: '$translation', conf: $confidence)")
                    serviceScope.launch {
                        val trans = translation ?: translationService.translateToEnglish(text, language)
                        _events.emit(PipelineUpdate.Transcript(text, trans, confidence))
                        val displayText = if (!trans.isNullOrBlank() && !trans.startsWith("Original")) "$text → $trans" else text
                        updateNotification(displayText)
                    }
                }

                override fun onError(error: String) {
                    Log.e(TAG, "onError: $error")
                    serviceScope.launch { _events.emit(PipelineUpdate.Error(error)) }
                }

                override fun onDone() {
                    Log.i(TAG, "onDone")
                    serviceScope.launch { _events.emit(PipelineUpdate.Done) }
                }
            })

            // Only initialize AudioRecordCapture when capturing internal speaker playback (MediaProjection)
            // to avoid competing with Google SODA / TPU SpeechRecognizer over the physical microphone
            if (hasProjection && mediaProjection != null) {
                audioCapture = AudioRecordCapture(
                    pipeline = handle,
                    sourceMode = sourceMode,
                    mediaProjection = mediaProjection,
                    disableMic = true,
                    onAudioPcm = { pcmBytes ->
                        audioPipeManager.write(pcmBytes)
                    }
                ).also { it.start() }
            }

            // Start Google Tensor TPU speech recognition
            startTpuSpeechRecognizer(language)

            serviceScope.launch {
                _events.emit(PipelineUpdate.Status("Listening: $sourceLabel"))
            }

        } catch (e: Throwable) {
            Log.e(TAG, "Startup failed: ${e.message}", e)
            serviceScope.launch { _events.emit(PipelineUpdate.Error("Startup failed: ${e.message}")) }
            stopService()
        }
    }

    private fun normalizeLocale(language: String): String = when (language.lowercase()) {
        "en", "auto", "und", "" -> "en-US"
        "fr" -> "fr-FR"
        "es" -> "es-ES"
        "de" -> "de-DE"
        "it" -> "it-IT"
        "ja" -> "ja-JP"
        "ko" -> "ko-KR"
        "zh" -> "zh-CN"
        "ar" -> "ar-SA"
        "ru" -> "ru-RU"
        "hi" -> "hi-IN"
        else -> if (language.contains("-")) language else "${language}-${language.uppercase()}"
    }

    private fun startTpuSpeechRecognizer(language: String) {
        mainHandler.post {
            if (!isListeningActive.get()) return@post

            val langTag = normalizeLocale(language)
            val isOnDeviceAvailable = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                SpeechRecognizer.isOnDeviceRecognitionAvailable(this@SpeechRecognitionService)
            } else false

            Log.i(TAG, "Google Tensor TPU On-Device Recognition Available: $isOnDeviceAvailable [lang=$langTag]")

            var useOffline = (langTag == "en-US") && isOnDeviceAvailable

            fun createFreshRecognizer(offline: Boolean): SpeechRecognizer {
                return try {
                    if (offline && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && isOnDeviceAvailable) {
                        Log.i(TAG, "Creating On-Device (Tensor TPU) SpeechRecognizer")
                        SpeechRecognizer.createOnDeviceSpeechRecognizer(this@SpeechRecognitionService)
                    } else {
                        Log.i(TAG, "Creating Standard SpeechRecognizer")
                        SpeechRecognizer.createSpeechRecognizer(this@SpeechRecognitionService)
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "Fallback to standard SpeechRecognizer: ${t.message}")
                    SpeechRecognizer.createSpeechRecognizer(this@SpeechRecognitionService)
                }
            }

            try {
                speechRecognizer?.cancel()
                speechRecognizer?.destroy()
            } catch (_: Throwable) {}

            var recognizer = createFreshRecognizer(offline = useOffline)
            speechRecognizer = recognizer

            fun buildRecognizerIntent(): Intent {
                return Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, langTag)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, langTag)
                    putExtra("android.speech.extra.DICTATION_MODE", true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 5000L)
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 4000L)

                    if (useOffline) {
                        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                    }

                    if (audioCapture != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        val pfd = audioPipeManager.createPipe()
                        if (pfd != null) {
                            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pfd)
                            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000)
                        }
                    }
                }
            }

            var consecutiveErrors = 0

            val listener = object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    Log.i(TAG, "Google SpeechRecognizer ready [lang=$langTag, offline=$useOffline]")
                    consecutiveErrors = 0
                }

                override fun onBeginningOfSpeech() {
                    Log.d(TAG, "Google SpeechRecognizer speech beginning")
                    serviceScope.launch { _events.emit(PipelineUpdate.Vad(1.0f, true)) }
                }

                override fun onRmsChanged(rmsdB: Float) {
                    val prob = ((rmsdB + 2.0f) / 10.0f).coerceIn(0.0f, 1.0f)
                    if (prob > 0.15f) {
                        serviceScope.launch { _events.emit(PipelineUpdate.Vad(prob, true)) }
                    }
                }

                override fun onBufferReceived(buffer: ByteArray?) {}

                override fun onEndOfSpeech() {
                    Log.d(TAG, "Google SpeechRecognizer speech end")
                }

                override fun onError(error: Int) {
                    val isSilence = error == SpeechRecognizer.ERROR_NO_MATCH ||
                                    error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT
                    if (!isSilence) {
                        consecutiveErrors++
                    }
                    val errorName = when (error) {
                        SpeechRecognizer.ERROR_AUDIO -> "ERROR_AUDIO"
                        SpeechRecognizer.ERROR_CLIENT -> "ERROR_CLIENT"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "ERROR_INSUFFICIENT_PERMISSIONS"
                        SpeechRecognizer.ERROR_NETWORK -> "ERROR_NETWORK"
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "ERROR_NETWORK_TIMEOUT"
                        SpeechRecognizer.ERROR_NO_MATCH -> "ERROR_NO_MATCH"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "ERROR_RECOGNIZER_BUSY"
                        SpeechRecognizer.ERROR_SERVER -> "ERROR_SERVER"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "ERROR_SPEECH_TIMEOUT"
                        12 -> "ERROR_LANGUAGE_NOT_SUPPORTED"
                        13 -> "ERROR_LANGUAGE_UNAVAILABLE"
                        14 -> "ERROR_CANNOT_CHECK_SUPPORT"
                        else -> "ERROR_$error"
                    }
                    Log.d(TAG, "Google SpeechRecognizer: $errorName ($error) [consecutive=$consecutiveErrors, isSilence=$isSilence]")

                    if (error in 12..14) {
                        Log.w(TAG, "Language $langTag offline pack missing on TPU; falling back to standard recognizer")
                        useOffline = false
                    }

                    if (!isListeningActive.get()) return

                    val delayMs = when {
                        isSilence -> 50L
                        error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> 250L
                        error == SpeechRecognizer.ERROR_CLIENT -> 150L
                        error in 12..14 -> 150L
                        else -> 100L
                    }

                    mainHandler.postDelayed({
                        if (!isListeningActive.get()) return@postDelayed
                        try {
                            if (consecutiveErrors >= 3 || error in 12..14) {
                                consecutiveErrors = 0
                                audioPipeManager.close()
                                speechRecognizer?.cancel()
                                speechRecognizer?.destroy()
                                val fresh = createFreshRecognizer(offline = useOffline)
                                fresh.setRecognitionListener(this)
                                speechRecognizer = fresh
                                fresh.startListening(buildRecognizerIntent())
                            } else {
                                audioPipeManager.close()
                                speechRecognizer?.cancel()
                                speechRecognizer?.startListening(buildRecognizerIntent())
                            }
                        } catch (t: Throwable) {
                            Log.e(TAG, "Error restarting SpeechRecognizer: ${t.message}")
                        }
                    }, delayMs)
                }

                override fun onResults(results: Bundle?) {
                    consecutiveErrors = 0
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = matches?.firstOrNull()?.trim()
                    if (!text.isNullOrEmpty()) {
                        hasTpuTranscript.set(true)
                        Log.i(TAG, "Google Speech Final Transcript: '$text'")
                        val cleaned = try {
                            pipeline?.cleanText(text) ?: text
                        } catch (_: Throwable) { text }

                        serviceScope.launch {
                            val trans = translationService.translateToEnglish(cleaned, langTag)
                            _events.emit(PipelineUpdate.Transcript(cleaned, trans, 0.99f))
                            val displayText = if (!trans.isNullOrBlank() && !trans.startsWith("Original")) "$cleaned → $trans" else cleaned
                            updateNotification(displayText)
                        }
                    }

                    if (isListeningActive.get()) {
                        mainHandler.postDelayed({
                            if (isListeningActive.get()) {
                                try {
                                    audioPipeManager.close()
                                    speechRecognizer?.cancel()
                                    speechRecognizer?.startListening(buildRecognizerIntent())
                                } catch (t: Throwable) {
                                    Log.e(TAG, "Error looping SpeechRecognizer: ${t.message}")
                                }
                            }
                        }, 50L)
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = matches?.firstOrNull()?.trim()
                    if (!text.isNullOrEmpty()) {
                        hasTpuTranscript.set(true)
                        Log.i(TAG, "Google Speech Partial Transcript: '$text'")
                        serviceScope.launch {
                            val trans = translationService.translateToEnglish(text, langTag)
                            _events.emit(PipelineUpdate.Transcript(text, trans, 0.85f))
                        }
                    }
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            }

            recognizer.setRecognitionListener(listener)
            try {
                recognizer.startListening(buildRecognizerIntent())
                Log.i(TAG, "Google SpeechRecognizer startListening initiated [lang=$langTag, offline=$useOffline]")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed initiating startListening: ${t.message}", t)
            }
        }
    }

    private fun buildNotification(contentText: String): Notification {
        val stopIntent = Intent(this, SpeechRecognitionService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Edge ORT Live Audio")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotification(text: String) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(text))
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Speech Recognition Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows status of real-time on-device speech transcription"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun stopService() {
        Log.i(TAG, "stopService executed")
        isListeningActive.set(false)
        _isRunning.value = false
        mainHandler.post {
            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.destroy()
            } catch (_: Throwable) {}
            speechRecognizer = null
        }

        audioPipeManager.close()
        audioCapture?.stop()
        audioCapture = null

        pipeline?.stop()
        pipeline = null

        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy called on SpeechRecognitionService")
        super.onDestroy()
        stopService()
        serviceScope.cancel()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
