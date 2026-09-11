package com.edgeort.android.download

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Background WorkManager worker that downloads on-device AI models reliably over Wi-Fi
 * or cellular, computing SHA-256 on the fly and writing atomically to disk.
 */
class ModelDownloadWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        const val KEY_URL = "model_url"
        const val KEY_DEST_PATH = "dest_path"
        const val KEY_EXPECTED_SHA256 = "expected_sha256"
        const val KEY_MODEL_NAME = "model_name"

        const val PROGRESS_PERCENT = "progress_percent"
        const val PROGRESS_BYTES = "bytes_downloaded"
        const val PROGRESS_TOTAL = "bytes_total"
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val modelUrl = inputData.getString(KEY_URL)
            ?: return@withContext Result.failure(workDataOf("error" to "Missing model URL"))
        val destPath = inputData.getString(KEY_DEST_PATH)
            ?: return@withContext Result.failure(workDataOf("error" to "Missing destination path"))
        val expectedSha256 = inputData.getString(KEY_EXPECTED_SHA256)
        val modelName = inputData.getString(KEY_MODEL_NAME) ?: "model"

        val targetFile = File(destPath)
        val parentDir = targetFile.parentFile
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs()
        }

        val tempFile = File("${destPath}.tmp")

        try {
            val url = URL(modelUrl)
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.instanceFollowRedirects = true

            if (connection.responseCode !in 200..299) {
                return@withContext Result.retry()
            }

            val totalBytes = connection.contentLengthLong
            var downloadedBytes = 0L

            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(64 * 1024) // 64 KB buffer

            connection.inputStream.use { input: InputStream ->
                FileOutputStream(tempFile).use { output: FileOutputStream ->
                    var bytesRead: Int
                    var lastReportedPercent = -1

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (isStopped) {
                            tempFile.delete()
                            return@withContext Result.failure(workDataOf("error" to "Cancelled"))
                        }

                        output.write(buffer, 0, bytesRead)
                        digest.update(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead

                        if (totalBytes > 0) {
                            val percent = ((downloadedBytes * 100) / totalBytes).toInt()
                            if (percent != lastReportedPercent) {
                                lastReportedPercent = percent
                                setProgress(
                                    workDataOf(
                                        PROGRESS_PERCENT to percent,
                                        PROGRESS_BYTES to downloadedBytes,
                                        PROGRESS_TOTAL to totalBytes
                                    )
                                )
                            }
                        }
                    }
                }
            }

            // Verify checksum if expectedSha256 is provided
            if (!expectedSha256.isNullOrBlank()) {
                val computedHash = digest.digest().joinToString("") { "%02x".format(it) }
                if (!computedHash.equals(expectedSha256.trim(), ignoreCase = true)) {
                    tempFile.delete()
                    return@withContext Result.failure(
                        workDataOf("error" to "SHA-256 mismatch for $modelName. Expected: $expectedSha256, got: $computedHash")
                    )
                }
            }

            // Atomically rename temp file to target destination
            if (targetFile.exists()) {
                targetFile.delete()
            }
            if (!tempFile.renameTo(targetFile)) {
                tempFile.copyTo(targetFile, overwrite = true)
                tempFile.delete()
            }

            Result.success(workDataOf("path" to targetFile.absolutePath, "model_name" to modelName))
        } catch (e: Exception) {
            if (tempFile.exists()) {
                tempFile.delete()
            }
            if (runAttemptCount < 3) {
                Result.retry()
            } else {
                Result.failure(workDataOf("error" to (e.message ?: "Download failed")))
            }
        }
    }
}
