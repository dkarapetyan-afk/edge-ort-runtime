package com.edgeort.android.download

import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

/**
 * Metadata descriptor for an on-device AI model asset.
 */
data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val downloadUrl: String,
    val fileName: String,
    val sha256Checksum: String? = null,
    val approximateSizeMb: Double = 0.0,
    val category: ModelCategory = ModelCategory.ASR
)

enum class ModelCategory {
    VAD,
    ASR,
    TRANSLATION,
    SLM
}

sealed class ModelDownloadStatus {
    object NotDownloaded : ModelDownloadStatus()
    data class Downloading(val progressPercent: Int, val bytesDownloaded: Long, val totalBytes: Long) : ModelDownloadStatus()
    data class Downloaded(val localPath: String) : ModelDownloadStatus()
    data class Error(val message: String) : ModelDownloadStatus()
}

/**
 * Centralized manager for background model downloads via Android WorkManager.
 * Exposes live status flows for Jetpack Compose UI updates and pipeline initialization.
 */
class ModelDownloadManager(
    private val context: Context,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Main)
) {
    private val workManager = WorkManager.getInstance(context)
    private val statusFlows = mutableMapOf<String, MutableStateFlow<ModelDownloadStatus>>()

    companion object {
        // Standard presets for Edge ORT mobile runtime
        val SILERO_VAD = ModelDescriptor(
            id = "silero_vad_v5",
            displayName = "Silero VAD v5 (INT8)",
            downloadUrl = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
            fileName = "silero_vad.onnx",
            sha256Checksum = null,
            approximateSizeMb = 1.8,
            category = ModelCategory.VAD
        )

        val WHISPER_TINY_EN = ModelDescriptor(
            id = "whisper_tiny_en_int8",
            displayName = "Whisper Tiny English (INT8)",
            downloadUrl = "https://huggingface.co/openai/whisper-tiny.en/resolve/main/onnx/model_quantized.onnx",
            fileName = "whisper_tiny_en_int8.onnx",
            sha256Checksum = null,
            approximateSizeMb = 39.0,
            category = ModelCategory.ASR
        )

        val OPUS_MT_EN_ES = ModelDescriptor(
            id = "opus_mt_en_es_int8",
            displayName = "Opus-MT English to Spanish (INT8)",
            downloadUrl = "https://huggingface.co/Helsinki-NLP/opus-mt-en-es/resolve/main/onnx/decoder_model_merged_quantized.onnx",
            fileName = "opus_mt_en_es_int8.onnx",
            sha256Checksum = null,
            approximateSizeMb = 55.0,
            category = ModelCategory.TRANSLATION
        )

        val SMOLLM_135M = ModelDescriptor(
            id = "smollm_135m_instruct_int4",
            displayName = "SmolLM 135M Instruct (INT4)",
            downloadUrl = "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/resolve/main/onnx/model_int4.onnx",
            fileName = "smollm_135m_instruct_int4.onnx",
            sha256Checksum = null,
            approximateSizeMb = 85.0,
            category = ModelCategory.SLM
        )

        val ALL_PRESETS = listOf(SILERO_VAD, WHISPER_TINY_EN, OPUS_MT_EN_ES, SMOLLM_135M)
    }

    /**
     * Get directory where models are stored on device.
     */
    fun getModelsDirectory(): File {
        val dir = File(context.filesDir, "models")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    /**
     * Get local file target for a model descriptor.
     */
    fun getModelFile(descriptor: ModelDescriptor): File {
        return File(getModelsDirectory(), descriptor.fileName)
    }

    /**
     * Check whether model file exists and is non-empty.
     */
    fun isModelDownloaded(descriptor: ModelDescriptor): Boolean {
        val file = getModelFile(descriptor)
        return file.exists() && file.length() > 1024
    }

    /**
     * Get or create a live StateFlow for observing model download progress.
     */
    @Synchronized
    fun getStatus(descriptor: ModelDescriptor): StateFlow<ModelDownloadStatus> {
        val flow = statusFlows.getOrPut(descriptor.id) {
            val initial = if (isModelDownloaded(descriptor)) {
                ModelDownloadStatus.Downloaded(getModelFile(descriptor).absolutePath)
            } else {
                ModelDownloadStatus.NotDownloaded
            }
            MutableStateFlow(initial)
        }

        // Re-check current disk state
        if (isModelDownloaded(descriptor) && flow.value !is ModelDownloadStatus.Downloaded) {
            flow.value = ModelDownloadStatus.Downloaded(getModelFile(descriptor).absolutePath)
        }

        return flow.asStateFlow()
    }

    /**
     * Enqueue a background download task using WorkManager.
     */
    fun startDownload(descriptor: ModelDescriptor, requireWifiOnly: Boolean = true): UUID {
        val targetFile = getModelFile(descriptor)
        val flow = statusFlows.getOrPut(descriptor.id) {
            MutableStateFlow(ModelDownloadStatus.NotDownloaded)
        }

        val constraints = Constraints.Builder()
            .setRequiredNetworkType(if (requireWifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
            .setRequiresStorageNotLow(true)
            .build()

        val inputData = workDataOf(
            ModelDownloadWorker.KEY_URL to descriptor.downloadUrl,
            ModelDownloadWorker.KEY_DEST_PATH to targetFile.absolutePath,
            ModelDownloadWorker.KEY_EXPECTED_SHA256 to descriptor.sha256Checksum,
            ModelDownloadWorker.KEY_MODEL_NAME to descriptor.displayName
        )

        val workRequest = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setConstraints(constraints)
            .setInputData(inputData)
            .addTag("model_download_${descriptor.id}")
            .build()

        workManager.enqueueUniqueWork(
            "download_${descriptor.id}",
            ExistingWorkPolicy.REPLACE,
            workRequest
        )

        flow.value = ModelDownloadStatus.Downloading(0, 0, 0)

        // Observe WorkManager status in scope
        scope.launch {
            workManager.getWorkInfoByIdFlow(workRequest.id).collect { workInfo ->
                if (workInfo != null) {
                    when (workInfo.state) {
                        WorkInfo.State.RUNNING -> {
                            val percent = workInfo.progress.getInt(ModelDownloadWorker.PROGRESS_PERCENT, 0)
                            val downloaded = workInfo.progress.getLong(ModelDownloadWorker.PROGRESS_BYTES, 0L)
                            val total = workInfo.progress.getLong(ModelDownloadWorker.PROGRESS_TOTAL, 0L)
                            flow.value = ModelDownloadStatus.Downloading(percent, downloaded, total)
                        }
                        WorkInfo.State.SUCCEEDED -> {
                            flow.value = ModelDownloadStatus.Downloaded(targetFile.absolutePath)
                        }
                        WorkInfo.State.FAILED -> {
                            val err = workInfo.outputData.getString("error") ?: "Download failed"
                            flow.value = ModelDownloadStatus.Error(err)
                        }
                        WorkInfo.State.CANCELLED -> {
                            flow.value = ModelDownloadStatus.NotDownloaded
                        }
                        else -> { /* ENQUEUED / BLOCKED */ }
                    }
                }
            }
        }

        return workRequest.id
    }

    /**
     * Cancel an active download.
     */
    fun cancelDownload(descriptor: ModelDescriptor) {
        workManager.cancelUniqueWork("download_${descriptor.id}")
        val flow = statusFlows[descriptor.id]
        if (flow != null && flow.value is ModelDownloadStatus.Downloading) {
            flow.value = ModelDownloadStatus.NotDownloaded
        }
    }

    /**
     * Delete downloaded model file from device.
     */
    fun deleteModel(descriptor: ModelDescriptor): Boolean {
        cancelDownload(descriptor)
        val file = getModelFile(descriptor)
        val deleted = if (file.exists()) file.delete() else false
        val flow = statusFlows[descriptor.id]
        if (flow != null) {
            flow.value = ModelDownloadStatus.NotDownloaded
        }
        return deleted
    }
}
