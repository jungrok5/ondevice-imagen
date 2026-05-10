package com.jungrok5.drawmoment

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Phase 3 entry point: the same SD 1.5 inference that
 * InferenceForegroundService runs, wrapped as a CoroutineWorker so the
 * weekly-postcard schedule can use WorkManager constraints
 * (charging + idle + battery-not-low + 7-day delay).
 *
 * Today: enqueued as OneTimeWork from the user-tap path.
 * Phase 3-3 (next): PeriodicWorkRequest with the constraints above.
 *
 * The Worker uploads itself to a foreground job via setForeground() so
 * Android 12+ keeps it alive across the full 5–15 min run; without
 * that, OS scheduling will pause us mid-inference.
 *
 * Notification channel IDs intentionally match
 * InferenceForegroundService — both code paths share them so users
 * never see "two apps" in their notification settings.
 */
class PostcardWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val prompt = inputData.getString(KEY_PROMPT) ?: ""
        val outputPath = inputData.getString(KEY_OUTPUT_PATH) ?: ""

        ensureChannels()
        // Promote to foreground BEFORE the long-running call. WorkManager
        // gives expedited workers a 10-min OS budget; for SDXL/SD 1.5 we
        // need the full ongoing-notification path instead.
        setForeground(buildForegroundInfo(prompt))

        return withContext(Dispatchers.IO) {
            try {
                runInference(prompt, outputPath)
                postCompletedNotification(outputPath)
                DrawMomentPlugin.liveInstance?.emitCompleted(outputPath)
                Result.success()
            } catch (e: Exception) {
                Log.e(TAG, "doWork: inference failed", e)
                postFailedNotification(e.message ?: "unknown error")
                DrawMomentPlugin.liveInstance?.emitFailed(e.message ?: "unknown error")
                // Result.failure: don't retry on its own. A periodic schedule
                // will pick up next cycle; the user-tap path surfaces the
                // error notification + signal already.
                Result.failure()
            }
        }
    }

    /** Body identical to InferenceForegroundService.doInference — picks
     *  fp16 bundle if pushed, else fp32, then runs SdInferencePipeline. */
    private fun runInference(prompt: String, outputPath: String) {
        val ext = applicationContext.getExternalFilesDir(null)
        val fp16Dir = File(ext, "onnx_fp16_unet/sd15_drawing_nty_scale0.8")
        val fp32Dir = File(ext, "onnx/sd15_drawing_nty_scale0.8")
        val (baseDir, isFp16) = when {
            fp16Dir.exists() -> fp16Dir to true
            fp32Dir.exists() -> fp32Dir to false
            else -> {
                Log.e(TAG, "model bundle missing at $fp16Dir or $fp32Dir — push via adb first")
                throw IllegalStateException("model bundle missing")
            }
        }
        Log.d(TAG, "runInference: bundle=${baseDir.absolutePath} fp16Unet=$isFp16")
        val pipe = SdInferencePipeline(baseDir = baseDir, unetIsFp16 = isFp16)
        val ms = pipe.generate(prompt, outputPath)
        Log.d(TAG, "runInference done in $ms ms — wrote $outputPath")
    }

    private fun buildForegroundInfo(prompt: String): ForegroundInfo {
        val n = NotificationCompat.Builder(applicationContext, CHANNEL_PROGRESS)
            .setContentTitle("Drawing this moment...")
            .setContentText(prompt.take(80))
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+ requires the foregroundServiceType bitfield at
            // the ForegroundInfo level too — same reason the manifest
            // alone wasn't enough for InferenceForegroundService on S25.
            ForegroundInfo(
                NOTIFICATION_ID_PROGRESS,
                n,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
            )
        } else {
            ForegroundInfo(NOTIFICATION_ID_PROGRESS, n)
        }
    }

    private fun ensureChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = applicationContext
                .getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(NotificationChannel(
                CHANNEL_PROGRESS, "Drawing in progress",
                NotificationManager.IMPORTANCE_LOW,
            ))
            nm.createNotificationChannel(NotificationChannel(
                CHANNEL_DONE, "Drawing finished",
                NotificationManager.IMPORTANCE_HIGH,
            ))
        }
    }

    private fun postCompletedNotification(outputPath: String) {
        val n = NotificationCompat.Builder(applicationContext, CHANNEL_DONE)
            .setContentTitle("Done — tap to view")
            .setContentText("Saved at ${File(outputPath).name}")
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentIntent(openAppPendingIntent())
            .setAutoCancel(true)
            .build()
        val nm = applicationContext
            .getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_DONE, n)
    }

    private fun postFailedNotification(errorMsg: String) {
        val n = NotificationCompat.Builder(applicationContext, CHANNEL_DONE)
            .setContentTitle("Drawing failed")
            .setContentText(errorMsg.take(120))
            .setSmallIcon(android.R.drawable.stat_notify_error)
            .setContentIntent(openAppPendingIntent())
            .setAutoCancel(true)
            .build()
        val nm = applicationContext
            .getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_DONE, n)
    }

    private fun openAppPendingIntent(): PendingIntent {
        val ctx = applicationContext
        val launchIntent = ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
        return PendingIntent.getActivity(
            ctx, 0, launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
    }

    companion object {
        const val KEY_PROMPT       = "prompt"
        const val KEY_OUTPUT_PATH  = "output_path"

        // Stable name so enqueueUniqueWork dedupes back-to-back taps —
        // a second tap while one is running gets dropped (KEEP policy).
        const val UNIQUE_NAME = "draw_moment_postcard"

        private const val TAG = "PostcardWorker"
        private const val CHANNEL_PROGRESS = "draw_moment_progress"
        private const val CHANNEL_DONE     = "draw_moment_done"
        private const val NOTIFICATION_ID_PROGRESS = 1001
        private const val NOTIFICATION_ID_DONE     = 1002
    }
}
