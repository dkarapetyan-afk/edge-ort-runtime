package com.edgeort.demo

import android.content.Context
import uniffi.edge_ort_runtime.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Example wrapper showing how Android Jetpack Compose or Foreground Services
 * consume the generated UniFFI edge_ort_runtime bindings.
 */
class AndroidSpeechPipeline(
    context: Context,
    languageCode: String = "auto",
    enableVad: Boolean = true,
    enableMt: Boolean = true
) {
    companion object {
        init {
            System.loadLibrary("edge_ort_runtime")
        }
    }

    private val pipeline: NativePipelineHandle = NativePipelineHandle(
        languageCode = languageCode,
        enableVad = enableVad,
        enableMt = enableMt,
        profilePath = context.filesDir.resolve("profiles/default").absolutePath
    )

    private val _liveTranscript = MutableStateFlow("")
    val liveTranscript = _liveTranscript.asStateFlow()

    private val _liveTranslation = MutableStateFlow<String?>(null)
    val liveTranslation = _liveTranslation.asStateFlow()

    private val _isSpeech = MutableStateFlow(false)
    val isSpeech = _isSpeech.asStateFlow()

    fun startListening() {
        pipeline.start(object : NativeSpeechListener {
            override fun onStatus(status: String) {
                println("Pipeline status: $status")
            }

            override fun onVad(speechProb: Float, isSpeechDetected: Boolean) {
                _isSpeech.value = isSpeechDetected
            }

            override fun onTranscript(text: String, translation: String?, confidence: Float) {
                _liveTranscript.value = text
                _liveTranslation.value = translation
            }

            override fun onError(error: String) {
                System.err.println("Pipeline error: $error")
            }

            override fun onDone() {
                println("Pipeline finished")
            }
        })
    }

    /**
     * Feed PCM audio directly from Google Oboe C++ callback or AudioRecord.
     * Automatically downmixes to mono and resamples to 16 kHz on the native side.
     */
    fun pushAudioBuffer(pcm16: ShortArray, sampleRate: Int = 48000, channels: Int = 1) {
        pipeline.pushPcm16Resampled(
            samples = pcm16.toList(),
            sampleRate = sampleRate.toUInt(),
            channels = channels.toUShort()
        )
    }

    fun stopListening() {
        pipeline.stop()
    }
}
