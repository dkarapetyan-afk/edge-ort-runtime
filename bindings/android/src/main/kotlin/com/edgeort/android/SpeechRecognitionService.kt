package com.edgeort.android

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import uniffi.edge_ort_runtime.*
import java.io.File

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
        const val EXTRA_LANGUAGE = "extra_language"
        const val EXTRA_AUDIO_SOURCE = "extra_audio_source"
        const val EXTRA_PROJECTION_RESULT_CODE = "extra_projection_result_code"
        const val EXTRA_PROJECTION_DATA = "extra_projection_data"

        private const val TAG = "SpeechRecService"

        // Event flow for activities and UI overlays
        private val _events = MutableSharedFlow<PipelineUpdate>(extraBufferCapacity = 64)
        val events: SharedFlow<PipelineUpdate> = _events.asSharedFlow()

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
        when (intent?.action) {
            ACTION_STOP -> {
                stopService()
                return START_NOT_STICKY
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
                    serviceScope.launch { _events.emit(PipelineUpdate.Status(status)) }
                    updateNotification(status)
                }

                override fun onVad(speechProb: Float, isSpeech: Boolean) {
                    serviceScope.launch { _events.emit(PipelineUpdate.Vad(speechProb, isSpeech)) }
                }

                override fun onTranscript(text: String, translation: String?, confidence: Float) {
                    serviceScope.launch {
                        _events.emit(PipelineUpdate.Transcript(text, translation, confidence))
                    }
                    val displayText = if (!translation.isNullOrBlank()) "$text → $translation" else text
                    updateNotification(displayText)
                }

                override fun onError(error: String) {
                    serviceScope.launch { _events.emit(PipelineUpdate.Error(error)) }
                }

                override fun onDone() {
                    serviceScope.launch { _events.emit(PipelineUpdate.Done) }
                }
            })

            // Start audio capture (Speakers + Mic or configured mode)
            audioCapture = AudioRecordCapture(
                pipeline = handle,
                sourceMode = sourceMode,
                mediaProjection = mediaProjection
            ).also { it.start() }

            serviceScope.launch {
                _events.emit(PipelineUpdate.Status("Listening: $sourceLabel"))
            }

        } catch (e: Throwable) {
            Log.e(TAG, "Startup failed: ${e.message}", e)
            serviceScope.launch { _events.emit(PipelineUpdate.Error("Startup failed: ${e.message}")) }
            stopService()
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
        super.onDestroy()
        stopService()
        serviceScope.cancel()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
