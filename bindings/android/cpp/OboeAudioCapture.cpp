#include "OboeAudioCapture.h"
#include <android/log.h>

#define LOG_TAG "EdgeOrtOboe"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace edge_ort {

OboeAudioCapture::OboeAudioCapture() = default;

OboeAudioCapture::~OboeAudioCapture() {
    stop();
}

bool OboeAudioCapture::start(const NativePipelineHandle* rustPipelineHandle, int32_t sampleRate) {
    if (isRecording_) {
        LOGI("OboeAudioCapture already recording");
        return true;
    }

    pipelineHandle_ = rustPipelineHandle;
    sampleRate_ = sampleRate;

    oboe::AudioStreamBuilder builder;
    builder.setDirection(oboe::Direction::Input)
           ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
           ->setSharingMode(oboe::SharingMode::Exclusive)
           ->setFormat(oboe::AudioFormat::I16)
           ->setChannelCount(oboe::ChannelCount::Mono)
           ->setSampleRate(sampleRate_)
           ->setInputPreset(oboe::InputPreset::VoiceRecognition)
           ->setDataCallback(this)
           ->setErrorCallback(this);

    oboe::Result result = builder.openStream(stream_);
    if (result != oboe::Result::OK) {
        LOGE("Failed to open Oboe audio stream: %s", oboe::convertToText(result));
        return false;
    }

    result = stream_->requestStart();
    if (result != oboe::Result::OK) {
        LOGE("Failed to start Oboe audio stream: %s", oboe::convertToText(result));
        stream_->close();
        stream_.reset();
        return false;
    }

    isRecording_ = true;
    LOGI("OboeAudioCapture started successfully at %d Hz", stream_->getSampleRate());
    return true;
}

void OboeAudioCapture::stop() {
    if (!isRecording_) return;

    isRecording_ = false;
    if (stream_) {
        stream_->stop();
        stream_->close();
        stream_.reset();
    }
    pipelineHandle_ = nullptr;
    LOGI("OboeAudioCapture stopped");
}

bool OboeAudioCapture::isRecording() const {
    return isRecording_;
}

oboe::DataCallbackResult OboeAudioCapture::onAudioReady(
    oboe::AudioStream * /*oboeStream*/,
    void *audioData,
    int32_t numFrames
) {
    if (!isRecording_ || !pipelineHandle_ || numFrames <= 0) {
        return oboe::DataCallbackResult::Continue;
    }

    const int16_t* pcm16 = static_cast<const int16_t*>(audioData);

    // Direct C-ABI call into Rust; handles downmixing & resampling if needed
    edge_ort_push_pcm16(
        pipelineHandle_,
        pcm16,
        static_cast<size_t>(numFrames),
        static_cast<uint32_t>(sampleRate_),
        1 // Mono
    );

    return oboe::DataCallbackResult::Continue;
}

void OboeAudioCapture::onErrorAfterClose(oboe::AudioStream* /*oboeStream*/, oboe::Result result) {
    LOGE("Oboe error occurred after close: %s", oboe::convertToText(result));
    isRecording_ = false;
}

} // namespace edge_ort
