#pragma once

#include <oboe/Oboe.h>
#include <memory>
#include <cstdint>

// Forward declaration of the opaque Rust pipeline handle
struct NativePipelineHandle;

// Direct Rust C-ABI export
extern "C" {
    int32_t edge_ort_push_pcm16(
        const NativePipelineHandle* handle,
        const int16_t* samples,
        size_t num_samples,
        uint32_t sample_rate,
        uint16_t channels
    );
}

namespace edge_ort {

/**
 * Low-latency audio capture using Google Oboe (AAudio/OpenSL ES).
 * Streams PCM buffers directly to the Rust Edge ORT pipeline via zero-copy C-ABI.
 */
class OboeAudioCapture : public oboe::AudioStreamDataCallback,
                         public oboe::AudioStreamErrorCallback {
public:
    OboeAudioCapture();
    ~OboeAudioCapture() override;

    bool start(const NativePipelineHandle* rustPipelineHandle, int32_t sampleRate = 48000);
    void stop();
    bool isRecording() const;

    // Oboe AudioStreamDataCallback
    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream *oboeStream,
        void *audioData,
        int32_t numFrames
    ) override;

    // Oboe AudioStreamErrorCallback
    void onErrorAfterClose(oboe::AudioStream *oboeStream, oboe::Result result) override;

private:
    std::shared_ptr<oboe::AudioStream> stream_;
    const NativePipelineHandle* pipelineHandle_{nullptr};
    int32_t sampleRate_{48000};
    bool isRecording_{false};
};

} // namespace edge_ort
