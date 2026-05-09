package com.jungrok5.drawmoment

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.IntBuffer
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
     * Phase 2 Step 5c — real on-device SD 1.5 inference.
     *
     * Pipeline (see SdInferencePipeline for details):
     *   1) tokenize prompt + "" through CLIP byte-level BPE
     *   2) text_encoder.run × 2 — cond + uncond hidden_states [1,77,768]
     *   3) Karras sigma schedule (12 steps, σ_max=14.6 → σ_min=0.029 → 0)
     *   4) gaussian latent at σ_max, deterministic seed = prompt.hashCode()
     *   5) 12-step Euler-Karras UNet loop, CFG=7.5 batch=2
     *   6) divide by 0.18215, vae_decoder.run → [-1,1] CHW float
     *   7) clamp + scale to uint8 ARGB → Bitmap → PNG
     *
     * Estimated wall clock on Note 10+ CPU EP: ~10 min (12 × ~50 s
     * per CFG-batched UNet step, plus negligible TE + VAE).
     */
    private fun doInference(prompt: String, outputPath: String) {
        val baseDir = File(
            getExternalFilesDir(null),
            "onnx/sd15_drawing_nty_scale0.8",
        )
        if (!baseDir.exists()) {
            Log.e(TAG, "model bundle missing at $baseDir — push it via adb first")
            throw IllegalStateException("model bundle missing")
        }
        val pipe = SdInferencePipeline(baseDir = baseDir)
        val ms = pipe.generate(prompt, outputPath)
        Log.d(TAG, "doInference done in $ms ms — wrote $outputPath")
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

    /** Phase 2 Step 5b-1 — encode a few prompts with the local
     *  ClipTokenizer and dump the IDs. Verified against
     *  transformers.CLIPTokenizer via scripts/clip_tokenizer_reference.py. */
    private fun probeTokenizer(livePrompt: String) {
        val tokDir = File(
            getExternalFilesDir(null),
            "onnx/sd15_drawing_nty_scale0.8/tokenizer",
        )
        Log.d(TAG, "tok: dir = $tokDir (exists=${tokDir.exists()})")
        if (!tokDir.exists()) {
            Log.w(TAG, "tok: tokenizer dir missing")
            return
        }
        val tok = try {
            ClipTokenizer(tokDir)
        } catch (e: Throwable) {
            Log.e(TAG, "tok: load FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
            return
        }
        val samples = listOf(
            "a photo of a cat",
            "a cat",
            "(style by NTY, drawing:1.2)",
            livePrompt,
        )
        for (s in samples) {
            val ids = tok.encode(s, ClipTokenizer.MAX_LENGTH)
            // Trim trailing pads for log clarity.
            val firstPad = ids.indexOfFirst { it == ids.last() && it == ids[ids.size - 1] }
            val nonPadEnd = (ids.size - 1 downTo 0).firstOrNull { ids[it] != ids[ids.size - 1] }
                ?.let { it + 1 } ?: 1
            val show = ids.copyOfRange(0, minOf(nonPadEnd + 1, 24))
            Log.d(TAG, "tok: '${s.take(60)}' -> ${show.joinToString(",")}" +
                if (nonPadEnd + 1 > 24) " ...(+more, total ${nonPadEnd + 1} non-pad)" else
                " (${nonPadEnd + 1} non-pad)"
            )
        }
    }

    /** Phase 2 Step 5b-4a — single forward pass through the UNet
     *  with deterministic inputs. Confirms (a) all 3 input tensors
     *  + 1 output tensor wire correctly, (b) inference completes
     *  without OOM, (c) one-step latency on Note 10+ CPU.
     *  Hardcoded prompt "a cat" so PC reference matches input. */
    private fun probeUNet() {
        val baseDir = File(
            getExternalFilesDir(null),
            "onnx/sd15_drawing_nty_scale0.8",
        )
        val tokDir = File(baseDir, "tokenizer")
        val ortEnv = OrtEnvironment.getEnvironment()

        // Encode the fixed prompt → input_ids → text_encoder → hidden_states
        val ids = ClipTokenizer(tokDir).encode("a cat", ClipTokenizer.MAX_LENGTH)
        val hiddenFlat: FloatArray
        run {
            var sess: OrtSession? = null
            var idsTensor: OnnxTensor? = null
            try {
                sess = ortEnv.createSession(
                    File(baseDir, "text_encoder/model.onnx").absolutePath,
                    OrtSession.SessionOptions(),
                )
                idsTensor = OnnxTensor.createTensor(
                    ortEnv, IntBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()),
                )
                val out = sess.run(mapOf("input_ids" to idsTensor))
                val hidden = out.get("last_hidden_state").get() as OnnxTensor
                val n = hidden.info.shape.fold(1L) { acc, d -> acc * d }.toInt()
                val flat = FloatArray(n)
                hidden.floatBuffer.get(flat)
                hidden.close()
                out.close()
                hiddenFlat = flat
            } finally {
                idsTensor?.close()
                sess?.close()
            }
        }

        // Build deterministic UNet inputs
        val sampleArr = FloatArray(1 * 4 * 64 * 64) { 0.5f }
        val timestepArr = floatArrayOf(999.0f)

        var unet: OrtSession? = null
        var sampleT: OnnxTensor? = null
        var tsT: OnnxTensor? = null
        var hiddenT: OnnxTensor? = null
        try {
            Log.d(TAG, "unet: opening UNet session...")
            val openT0 = System.currentTimeMillis()
            unet = ortEnv.createSession(
                File(baseDir, "unet/model.onnx").absolutePath,
                OrtSession.SessionOptions(),
            )
            Log.d(TAG, "unet: session open in ${System.currentTimeMillis() - openT0} ms")

            sampleT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(sampleArr), longArrayOf(1, 4, 64, 64),
            )
            tsT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(timestepArr), longArrayOf(1),
            )
            hiddenT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(hiddenFlat), longArrayOf(1, 77, 768),
            )

            Log.d(TAG, "unet: running 1 forward pass (sample=0.5, t=999, prompt='a cat')...")
            val runT0 = System.currentTimeMillis()
            val out = unet.run(mapOf(
                "sample" to sampleT,
                "timestep" to tsT,
                "encoder_hidden_states" to hiddenT,
            ))
            val elapsed = System.currentTimeMillis() - runT0

            val outSample = out.get("out_sample").get() as OnnxTensor
            val outShape = outSample.info.shape
            val n = outShape.fold(1L) { acc, d -> acc * d }.toInt()
            val outFlat = FloatArray(n)
            outSample.floatBuffer.get(outFlat)
            outSample.close()
            out.close()

            // Stats
            var sum = 0.0
            var sumSq = 0.0
            var mn = Float.POSITIVE_INFINITY
            var mx = Float.NEGATIVE_INFINITY
            for (v in outFlat) {
                sum += v; sumSq += v.toDouble() * v.toDouble()
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            val mean = sum / n
            val variance = sumSq / n - mean * mean
            val stddev = if (variance > 0) Math.sqrt(variance) else 0.0
            Log.d(TAG, "unet: shape=${outShape.joinToString("x")} elapsed=${elapsed} ms")
            Log.d(TAG, "unet: stats mean=${"%.5f".format(mean)} std=${"%.5f".format(stddev)} " +
                "min=${"%.5f".format(mn)} max=${"%.5f".format(mx)}")
            Log.d(TAG, "unet: out[0,0,0,0..7]=" +
                (0..7).joinToString(",") { "%.4f".format(outFlat[it]) })
        } catch (e: Throwable) {
            Log.e(TAG, "unet: FAILED ${e.javaClass.simpleName}: ${e.message}", e)
        } finally {
            sampleT?.close()
            tsT?.close()
            hiddenT?.close()
            unet?.close()
        }
    }

    /** Phase 2 Step 5b-3a/b — emit Karras sigma + timestep schedule
     *  AND a single canned Euler step for cross-check against
     *  scripts/scheduler_reference.py. */
    private fun probeScheduler() {
        val sched = DpmScheduler()
        val sigmas = sched.karrasSigmas(numInferenceSteps = 12)
        val ts = sched.karrasTimesteps(sigmas)
        Log.d(TAG, "sched: sigmas n+1=${sigmas.size} (first/last) " +
            "${"%.6f".format(sigmas[0])}, ${"%.6f".format(sigmas.last())}")
        Log.d(TAG, "sched: timesteps = ${ts.joinToString(",")}")

        // Canned Euler step against the same dummy values as the
        // PC reference. Both arrays come from
        // numpy.random.RandomState(42).randn(1,4,2,2) — 16 floats.
        // Hardcoding lets Kotlin match PC exactly without porting RNG.
        val dummySample = floatArrayOf(
            0.49671414f, -0.1382643f, 0.6476886f, 1.5230298f,    // [0,0]
            -0.23413695f, -0.23415337f, 1.5792128f, 0.7674347f,  // [0,1]
            -0.46947438f, 0.54256004f, -0.46341768f, -0.46572974f, // [0,2]
            0.24196227f, -1.9132802f, -1.7249179f, -0.5622875f,    // [0,3]
        )
        val dummyNoise = floatArrayOf(
            -0.50641555f, 0.15712367f, -0.45401204f, -0.7061518f,
            0.3509454f, -0.0511145f, -0.46428713f, -0.27044722f,
            0.5251812f, -0.0925275f, -0.3144878f, 0.07989023f,
            -0.10894683f, 0.04509933f, 0.7038389f, 0.18338153f,
        )
        // Step at index 0: sigma_t=sigmas[0], sigma_s=sigmas[1]
        val out = sched.stepEuler(dummySample, dummyNoise, sigmas[0], sigmas[1])
        Log.d(TAG, "sched: step(idx=0, sigma_t=${"%.4f".format(sigmas[0])} -> " +
            "sigma_s=${"%.4f".format(sigmas[1])}) prev[0,0,0..3]=" +
            (0..3).joinToString(",") { "%.4f".format(out[it]) })
        Log.d(TAG, "sched: stats prev mean=${"%.5f".format(out.average())} " +
            "min=${"%.5f".format(out.min())} max=${"%.5f".format(out.max())}")
    }

    /** Phase 2 Step 5b-2 — tokenize + text_encoder.run, dump shape +
     *  summary stats. Cross-check via scripts/text_encoder_reference.py. */
    private fun probeTextEncoder(prompt: String) {
        val baseDir = File(
            getExternalFilesDir(null),
            "onnx/sd15_drawing_nty_scale0.8",
        )
        val tokDir = File(baseDir, "tokenizer")
        val tok = try {
            ClipTokenizer(tokDir)
        } catch (e: Throwable) {
            Log.e(TAG, "te: tokenizer load FAILED: ${e.message}", e)
            return
        }
        val ids = tok.encode(prompt, ClipTokenizer.MAX_LENGTH)
        Log.d(TAG, "te: ids[0..7]=${ids.take(8)} ids[-3..]=${ids.takeLast(3)}")

        val ortEnv = OrtEnvironment.getEnvironment()
        var session: OrtSession? = null
        var inputTensor: OnnxTensor? = null
        try {
            session = ortEnv.createSession(
                File(baseDir, "text_encoder/model.onnx").absolutePath,
                OrtSession.SessionOptions(),
            )
            // CLIP text_encoder takes input_ids as INT32 [batch=1, seq=77].
            // ORT Java's IntBuffer overload picks INT32 automatically.
            val buf = IntBuffer.wrap(ids)
            inputTensor = OnnxTensor.createTensor(
                ortEnv, buf, longArrayOf(1, ids.size.toLong()),
            )
            val t0 = System.currentTimeMillis()
            val out = session.run(mapOf("input_ids" to inputTensor))
            val elapsed = System.currentTimeMillis() - t0
            val hidden = out.get("last_hidden_state").get() as OnnxTensor
            val shape = hidden.info.shape    // [1, 77, 768]
            val flat = FloatArray(shape.fold(1L) { acc, d -> acc * d }.toInt())
            hidden.floatBuffer.get(flat)
            // Stats — mean, stddev, min, max, plus a fingerprint of the
            // first 8 values so any drift vs PC reference is visible.
            val n = flat.size
            var sum = 0.0
            var sumSq = 0.0
            var mn = Float.POSITIVE_INFINITY
            var mx = Float.NEGATIVE_INFINITY
            for (v in flat) {
                sum += v; sumSq += v.toDouble() * v.toDouble()
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            val mean = sum / n
            val variance = sumSq / n - mean * mean
            val stddev = if (variance > 0) Math.sqrt(variance) else 0.0
            Log.d(TAG, "te: shape=${shape.joinToString("x")} elapsed=${elapsed} ms")
            Log.d(TAG, "te: stats mean=${"%.5f".format(mean)} std=${"%.5f".format(stddev)} " +
                "min=${"%.5f".format(mn)} max=${"%.5f".format(mx)}")
            // First 8 values of the FIRST token's embedding (the BOS row)
            Log.d(TAG, "te: hid[0,0,0..7]=" +
                (0 until 8).joinToString(",") { "%.4f".format(flat[it]) })
            // First 8 values of the SECOND token's embedding (token 1)
            val rowOffset = 768
            Log.d(TAG, "te: hid[0,1,0..7]=" +
                (0 until 8).joinToString(",") { "%.4f".format(flat[rowOffset + it]) })
            hidden.close()
            out.close()
        } catch (e: Throwable) {
            Log.e(TAG, "te: run FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
        } finally {
            inputTensor?.close()
            session?.close()
        }
    }

    /** Phase 2 Step 5b-1 — encode a few prompts with the local
     *  ClipTokenizer and dump the IDs. Verified against
     *  transformers.CLIPTokenizer via scripts/clip_tokenizer_reference.py. */

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
