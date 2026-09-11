package com.edgeort.android

import com.edgeort.android.download.ModelCategory
import com.edgeort.android.download.ModelDescriptor
import com.edgeort.android.download.ModelDownloadManager
import com.edgeort.android.download.ModelDownloadStatus
import org.junit.Assert.*
import org.junit.Test

class ModelDownloadManagerTest {

    @Test
    fun testStandardPresetsValidation() {
        val presets = ModelDownloadManager.ALL_PRESETS
        assertEquals(4, presets.size)

        val silero = ModelDownloadManager.SILERO_VAD
        assertEquals(ModelCategory.VAD, silero.category)
        assertTrue(silero.fileName.endsWith(".onnx"))
        assertTrue(silero.approximateSizeMb > 0)

        val whisper = ModelDownloadManager.WHISPER_TINY_EN
        assertEquals(ModelCategory.ASR, whisper.category)
        assertTrue(whisper.downloadUrl.startsWith("https://"))

        val opus = ModelDownloadManager.OPUS_MT_EN_ES
        assertEquals(ModelCategory.TRANSLATION, opus.category)

        val smollm = ModelDownloadManager.SMOLLM_135M
        assertEquals(ModelCategory.SLM, smollm.category)
    }

    @Test
    fun testModelDownloadStatusStates() {
        val notDownloaded: ModelDownloadStatus = ModelDownloadStatus.NotDownloaded
        assertTrue(notDownloaded is ModelDownloadStatus.NotDownloaded)

        val downloading: ModelDownloadStatus = ModelDownloadStatus.Downloading(
            progressPercent = 45,
            bytesDownloaded = 4500L,
            totalBytes = 10000L
        )
        assertTrue(downloading is ModelDownloadStatus.Downloading)
        val dl = downloading as ModelDownloadStatus.Downloading
        assertEquals(45, dl.progressPercent)
        assertEquals(4500L, dl.bytesDownloaded)
        assertEquals(10000L, dl.totalBytes)

        val downloaded: ModelDownloadStatus = ModelDownloadStatus.Downloaded("/data/models/silero.onnx")
        assertEquals("/data/models/silero.onnx", (downloaded as ModelDownloadStatus.Downloaded).localPath)

        val error: ModelDownloadStatus = ModelDownloadStatus.Error("Network timeout")
        assertEquals("Network timeout", (error as ModelDownloadStatus.Error).message)
    }

    @Test
    fun testCustomDescriptorCreation() {
        val custom = ModelDescriptor(
            id = "custom_npu_model",
            displayName = "Custom Model",
            downloadUrl = "https://example.com/model.onnx",
            fileName = "custom.onnx",
            sha256Checksum = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            approximateSizeMb = 12.5,
            category = ModelCategory.ASR
        )
        assertEquals("custom_npu_model", custom.id)
        assertEquals("custom.onnx", custom.fileName)
        assertNotNull(custom.sha256Checksum)
    }
}
