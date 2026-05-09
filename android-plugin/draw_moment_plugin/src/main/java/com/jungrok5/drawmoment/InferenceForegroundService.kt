package com.jungrok5.drawmoment

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

/**
 * Long-running inference task.
 *
 * Phase 1 body: sleep 30 s → write a placeholder PNG → emit completed
 *               signal + replace the ongoing notification with a
 *               "Done — tap to view" one.
 * Phase 2: replace the sleep+placeholder block with a real ONNX
 *          Runtime SD 1.5 inference. Everything around it (foreground
 *          notification, completion notification, signal back to GDScript)
 *          stays.
 */
class InferenceForegroundService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val prompt = intent?.getStringExtra(EXTRA_PROMPT) ?: ""
        val outputPath = intent?.getStringExtra(EXTRA_OUTPUT_PATH) ?: ""

        ensureChannels()
        startForeground(NOTIFICATION_ID_PROGRESS, buildProgressNotification(prompt))

        thread(name = "inference-worker") {
            try {
                doInference(prompt, outputPath)
                postCompletedNotification(outputPath)
                DrawMomentPlugin.liveInstance?.emitCompleted(outputPath)
            } catch (e: Exception) {
                postFailedNotification(e.message ?: "unknown error")
                DrawMomentPlugin.liveInstance?.emitFailed(e.message ?: "unknown error")
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }

        // Restart-not-sticky: if the OS kills us mid-job we don't auto-restart.
        // The user has to press the button again. Cleaner than waking up
        // hours later with stale state.
        return START_NOT_STICKY
    }

    /**
     * Phase-1 placeholder body. Writes a tiny PNG with the prompt drawn
     * onto it after a 30 s wait. Phase 2 replaces this entire function
     * with ONNX Runtime SD 1.5 inference (and the wait drops to whatever
     * actual inference takes — 1–3 min on Note 10+).
     */
    private fun doInference(prompt: String, outputPath: String) {
        // Phase 2 Step 5a: probe each ONNX sub-model. Confirms ORT
        // loads, the model files are reachable from this app context,
        // and emits each session's input/output schemas to logcat so
        // we know what tensors the next milestone (5b) needs to feed.
        // Bundled at /sdcard/Android/data/<pkg>/files/onnx/<name>/
        // via `adb push` for now; download-on-first-use comes later.
        probeOnnxModels()

        Thread.sleep(30_000)

        val bmp = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.rgb(245, 240, 230))
        val paint = Paint().apply {
            color = Color.rgb(40, 40, 40)
            textSize = 22f
            isAntiAlias = true
        }
        canvas.drawText("PHASE 1 — fake inference", 20f, 40f, paint)
        paint.textSize = 14f
        // Word-wrap the prompt onto subsequent lines.
        var y = 80f
        for (chunk in prompt.chunked(48)) {
            canvas.drawText(chunk, 20f, y, paint)
            y += 22f
        }
        paint.color = Color.rgb(120, 80, 80)
        paint.textSize = 14f
        canvas.drawText("real SDXL/SD 1.5 output appears here in phase 2",
            20f, 500f, paint)

        FileOutputStream(File(outputPath)).use { out ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        bmp.recycle()
    }

    /**
     * Phase 2 Step 5a probe — open each SD sub-model in turn, log its
     * input/output schemas, close it. Confirms (a) ORT loads on the
     * device, (b) the bundle directory is reachable from app context,
     * (c) what shapes/dtypes Step 5b needs to feed into Session.run().
     *
     * UNet is loaded last + logged most carefully because it's the
     * 1.7 GB hot path; if it OOMs on Note 10+ we need to know before
     * 5b wires the full pipeline.
     */
    private fun probeOnnxModels() {
        val baseDir = File(
            getExternalFilesDir(null),
            "onnx/sd15_drawing_nty_scale0.8",
        )
        Log.d(TAG, "probe: bundle dir = $baseDir (exists=${baseDir.exists()})")
        if (!baseDir.exists()) {
            Log.w(TAG, "probe: bundle dir missing — push the ONNX export there first")
            return
        }

        val ortEnv = OrtEnvironment.getEnvironment()
        // text_encoder + VAEs are small enough to load+close trivially;
        // UNet last so its memory footprint dominates only briefly.
        val components = listOf("text_encoder", "vae_encoder", "vae_decoder", "unet")
        for (comp in components) {
            val modelFile = File(baseDir, "$comp/model.onnx")
            val sizeMb = if (modelFile.exists()) modelFile.length() / 1_000_000 else 0
            Log.d(TAG, "probe: $comp $modelFile (size=${sizeMb} MB)")
            if (!modelFile.exists()) {
                Log.w(TAG, "probe: $comp missing")
                continue
            }
            var session: OrtSession? = null
            try {
                val opts = OrtSession.SessionOptions()
                session = ortEnv.createSession(modelFile.absolutePath, opts)
                Log.d(TAG, "probe: $comp loaded — inputs=${session.inputNames}")
                Log.d(TAG, "probe: $comp           outputs=${session.outputNames}")
                for (name in session.inputNames) {
                    val info = session.inputInfo[name]
                    Log.d(TAG, "probe: $comp input '$name' info: $info")
                }
            } catch (e: Throwable) {
                // Throwable so OOM (Error subclass) also surfaces.
                Log.e(TAG, "probe: $comp load FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
            } finally {
                session?.close()
            }
        }
        Log.d(TAG, "probe: done")
    }

    // --- Notifications -------------------------------------------------------

    private fun ensureChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(NotificationChannel(
                CHANNEL_PROGRESS, "Drawing in progress",
                NotificationManager.IMPORTANCE_LOW
            ))
            nm.createNotificationChannel(NotificationChannel(
                CHANNEL_DONE, "Drawing finished",
                NotificationManager.IMPORTANCE_HIGH
            ))
        }
    }

    private fun buildProgressNotification(prompt: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_PROGRESS)
            .setContentTitle("Drawing this moment...")
            .setContentText(prompt.take(80))
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun postCompletedNotification(outputPath: String) {
        val pi = openAppPendingIntent()
        val n = NotificationCompat.Builder(this, CHANNEL_DONE)
            .setContentTitle("Done — tap to view")
            .setContentText("Saved at ${File(outputPath).name}")
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .build()
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_DONE, n)
    }

    private fun postFailedNotification(errorMsg: String) {
        val pi = openAppPendingIntent()
        val n = NotificationCompat.Builder(this, CHANNEL_DONE)
            .setContentTitle("Drawing failed")
            .setContentText(errorMsg.take(120))
            .setSmallIcon(android.R.drawable.stat_notify_error)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .build()
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_DONE, n)
    }

    private fun openAppPendingIntent(): PendingIntent {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
            ?: Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        launchIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
        return PendingIntent.getActivity(
            this, 0, launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
    }

    companion object {
        const val EXTRA_PROMPT       = "prompt"
        const val EXTRA_OUTPUT_PATH  = "output_path"

        private const val TAG = "InferenceFgService"
        private const val CHANNEL_PROGRESS = "draw_moment_progress"
        private const val CHANNEL_DONE     = "draw_moment_done"
        private const val NOTIFICATION_ID_PROGRESS = 1001
        private const val NOTIFICATION_ID_DONE     = 1002
    }
}
