package com.jungrok5.drawmoment

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.Random
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phase 2 Steps 5b-4b + 5b-5 + 5c — assemble the on-device SD 1.5
 * pipeline using the ORT sub-models, the Kotlin tokenizer, the
 * Karras+Euler scheduler, and standard CFG. Output is a 512x512 PNG
 * at `outputPath`.
 *
 * Roughly 600 s on Note 10+ CPU at 12 steps with CFG batch=2.
 *
 * Limitations:
 *   - fp32 only (Step 3 fp16 cast deferred — see quantize_onnx_step3.py)
 *   - Euler-Karras scheduler, not DPM++ 2M (port deferred — same reason)
 *   - CPU EP, no NNAPI/QNN (Step 6)
 *   - Single LoRA (the one baked into the fused checkpoint at PC merge)
 */
class SdInferencePipeline(
    private val baseDir: File,
    private val numInferenceSteps: Int = 12,
    private val cfgScale: Float = 7.5f,
    private val width: Int = 512,
    private val height: Int = 512,
) {
    private val latentH = height / 8
    private val latentW = width / 8
    private val ortEnv = OrtEnvironment.getEnvironment()

    /** Run the full pipeline and write a PNG to `outputPath`.
     *  Returns elapsed milliseconds. */
    fun generate(prompt: String, outputPath: String): Long {
        val totalT0 = System.currentTimeMillis()
        Log.d(TAG, "pipe: prompt='${prompt.take(60)}'")
        Log.d(TAG, "pipe: steps=$numInferenceSteps cfg=$cfgScale ${width}x${height}")

        // 1) Text encoding for both cond and uncond — feed the
        //    UNet as a CFG batch of 2.
        val tok = ClipTokenizer(File(baseDir, "tokenizer"))
        val condIds   = tok.encode(prompt, ClipTokenizer.MAX_LENGTH)
        val uncondIds = tok.encode("",     ClipTokenizer.MAX_LENGTH)

        val teSession = ortEnv.createSession(
            File(baseDir, "text_encoder/model.onnx").absolutePath,
            OrtSession.SessionOptions(),
        )
        val condHidden   = runTextEncoder(teSession, condIds)
        val uncondHidden = runTextEncoder(teSession, uncondIds)
        teSession.close()
        Log.d(TAG, "pipe: text_encoder done — cond mean=${"%.5f".format(condHidden.average())}")

        // 2) Karras sigma schedule + a deterministic latent at sigma_max.
        val sched = DpmScheduler()
        val sigmas = sched.karrasSigmas(numInferenceSteps)
        val timesteps = sched.karrasTimesteps(sigmas)
        val seed = prompt.hashCode().toLong()
        var latent = gaussianLatent(seed, sigmas[0])
        Log.d(TAG, "pipe: latent[0..3]=" +
            (0..3).joinToString(",") { "%.4f".format(latent[it]) } +
            " sigma0=${"%.3f".format(sigmas[0])}")

        // 3) UNet 12-step loop with CFG batch=2.
        //    EulerDiscreteScheduler.scale_model_input divides the
        //    sample by sqrt(sigma^2 + 1) before feeding it to UNet.
        //    Skipping this step (which we did the first time) makes
        //    the magnitudes wrong → noise_pred drifts → final image
        //    collapses to a flat color. The Euler step itself still
        //    runs on the *raw* (unscaled) latent.
        val unetSession = ortEnv.createSession(
            File(baseDir, "unet/model.onnx").absolutePath,
            OrtSession.SessionOptions(),
        )
        try {
            for (k in 0 until numInferenceSteps) {
                val sigmaCurrent = sigmas[k]
                val sigmaNext    = sigmas[k + 1]
                val scaleFactor  = 1f / sqrt(sigmaCurrent * sigmaCurrent + 1f)
                val scaledSample = FloatArray(latent.size) {
                    latent[it] * scaleFactor
                }

                val stepT0 = System.currentTimeMillis()
                val noisePred = runUnetCfg(
                    unetSession, scaledSample, timesteps[k].toFloat(),
                    condHidden, uncondHidden,
                )
                latent = sched.stepEuler(latent, noisePred, sigmaCurrent, sigmaNext)
                val stepDt = System.currentTimeMillis() - stepT0
                Log.d(TAG, "pipe: step ${k + 1}/$numInferenceSteps t=${timesteps[k]} " +
                    "sigma=${"%.3f".format(sigmaCurrent)}->${"%.3f".format(sigmaNext)} " +
                    "${stepDt}ms")
            }
        } finally {
            unetSession.close()
        }

        // 4) VAE decode the final latent.
        val vaeSession = ortEnv.createSession(
            File(baseDir, "vae_decoder/model.onnx").absolutePath,
            OrtSession.SessionOptions(),
        )
        val image: FloatArray
        try {
            // SD 1.5 latent scale — diffusers divides by 0.18215 before VAE.
            val scaled = FloatArray(latent.size) { latent[it] / 0.18215f }
            image = runVaeDecoder(vaeSession, scaled)
            Log.d(TAG, "pipe: vae_decoder done — image[0..3]=" +
                (0..3).joinToString(",") { "%.4f".format(image[it]) })
        } finally {
            vaeSession.close()
        }

        // 5) [-1,1] CHW float → [0,255] HWC uint8 → Bitmap → PNG.
        val bitmap = floatChwToBitmap(image, width, height)
        FileOutputStream(File(outputPath)).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        bitmap.recycle()

        val elapsed = System.currentTimeMillis() - totalT0
        Log.d(TAG, "pipe: PNG saved to $outputPath ($elapsed ms total)")
        return elapsed
    }

    // --- helpers ----------------------------------------------------------

    private fun runTextEncoder(session: OrtSession, ids: IntArray): FloatArray {
        var idsTensor: OnnxTensor? = null
        try {
            idsTensor = OnnxTensor.createTensor(
                ortEnv, IntBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()),
            )
            val out = session.run(mapOf("input_ids" to idsTensor))
            val hidden = out.get("last_hidden_state").get() as OnnxTensor
            val n = hidden.info.shape.fold(1L) { acc, d -> acc * d }.toInt()
            val flat = FloatArray(n)
            hidden.floatBuffer.get(flat)
            hidden.close()
            out.close()
            return flat
        } finally {
            idsTensor?.close()
        }
    }

    /** UNet forward with CFG: stack [uncond, cond] in a batch of 2,
     *  one UNet.run(), then split the noise predictions and combine. */
    private fun runUnetCfg(
        session: OrtSession,
        sample: FloatArray,
        timestep: Float,
        condHidden: FloatArray,
        uncondHidden: FloatArray,
    ): FloatArray {
        val perFrame = 4 * latentH * latentW
        require(sample.size == perFrame) { "sample size ${sample.size} != $perFrame" }

        // Concatenate sample twice along batch dim
        val sampleBatch = FloatArray(2 * perFrame)
        System.arraycopy(sample, 0, sampleBatch, 0, perFrame)
        System.arraycopy(sample, 0, sampleBatch, perFrame, perFrame)

        // Concatenate uncond + cond hidden states along batch dim
        val perHidden = 77 * 768
        val hiddenBatch = FloatArray(2 * perHidden)
        System.arraycopy(uncondHidden, 0, hiddenBatch, 0,        perHidden)
        System.arraycopy(condHidden,   0, hiddenBatch, perHidden, perHidden)

        val tsBatch = floatArrayOf(timestep, timestep)

        var sampleT: OnnxTensor? = null
        var tsT:     OnnxTensor? = null
        var hiddenT: OnnxTensor? = null
        try {
            sampleT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(sampleBatch),
                longArrayOf(2, 4, latentH.toLong(), latentW.toLong()),
            )
            tsT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(tsBatch), longArrayOf(2),
            )
            hiddenT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(hiddenBatch), longArrayOf(2, 77, 768),
            )
            val out = session.run(mapOf(
                "sample"               to sampleT,
                "timestep"             to tsT,
                "encoder_hidden_states" to hiddenT,
            ))
            val outSample = out.get("out_sample").get() as OnnxTensor
            val flat = FloatArray(2 * perFrame)
            outSample.floatBuffer.get(flat)
            outSample.close()
            out.close()

            // Split + classifier-free-guidance combine
            // noise = uncond + cfgScale * (cond - uncond)
            val combined = FloatArray(perFrame)
            for (i in 0 until perFrame) {
                val nUncond = flat[i]
                val nCond   = flat[perFrame + i]
                combined[i] = nUncond + cfgScale * (nCond - nUncond)
            }
            return combined
        } finally {
            sampleT?.close()
            tsT?.close()
            hiddenT?.close()
        }
    }

    private fun runVaeDecoder(session: OrtSession, latent: FloatArray): FloatArray {
        var latentT: OnnxTensor? = null
        try {
            latentT = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(latent),
                longArrayOf(1, 4, latentH.toLong(), latentW.toLong()),
            )
            val out = session.run(mapOf("latent_sample" to latentT))
            val image = out.get("sample").get() as OnnxTensor
            val n = image.info.shape.fold(1L) { acc, d -> acc * d }.toInt()
            val flat = FloatArray(n)
            image.floatBuffer.get(flat)
            image.close()
            out.close()
            return flat
        } finally {
            latentT?.close()
        }
    }

    /** Box-Muller deterministic gaussian latent, shape [1,4,H,W],
     *  scaled by sigma so the final-step Karras sample lands near
     *  data distribution after the schedule denoises. */
    private fun gaussianLatent(seed: Long, sigma: Float): FloatArray {
        val rng = Random(seed)
        val n = 4 * latentH * latentW
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            // Box-Muller: produces two independent N(0,1) samples per loop.
            val u1 = (rng.nextDouble() + 1e-12).coerceAtMost(1.0 - 1e-12)
            val u2 = rng.nextDouble()
            val mag = sqrt(-2.0 * ln(u1))
            val z0 = mag * cos(2.0 * Math.PI * u2)
            val z1 = mag * sin(2.0 * Math.PI * u2)
            out[i] = (z0 * sigma).toFloat()
            if (i + 1 < n) out[i + 1] = (z1 * sigma).toFloat()
            i += 2
        }
        return out
    }

    /** Convert UNet/VAE [B=1, C=3, H, W] FLOAT in [-1,1] to a Bitmap.
     *  Clamps + scales to [0,255] uint8, packs CHW → HWC. */
    private fun floatChwToBitmap(chw: FloatArray, w: Int, h: Int): Bitmap {
        val pixels = IntArray(w * h)
        val planeSize = w * h
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val r = clampU8(chw[0 * planeSize + idx])
                val g = clampU8(chw[1 * planeSize + idx])
                val b = clampU8(chw[2 * planeSize + idx])
                pixels[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun clampU8(v: Float): Int {
        // [-1,1] -> [0,255]
        val x = ((v + 1f) * 0.5f * 255f + 0.5f).toInt()
        return x.coerceIn(0, 255)
    }

    companion object {
        private const val TAG = "SdInferencePipeline"
    }
}
