package com.jungrok5.drawmoment

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
    /** Step 6-A: when the UNet sub-model was built with
     *  `keep_io_types=False` so its input/output tensors are fp16,
     *  we must hand it Float16 tensors and cast its output back to
     *  fp32 before the Euler step (which still runs in fp32). The
     *  text_encoder + VAE stay fp32 — only UNet is fp16. */
    private val unetIsFp16: Boolean = false,
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
            sessionOptions(),
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
            sessionOptions(),
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
            sessionOptions(),
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

    // --- Step 6-A fp16 helpers ------------------------------------------

    /** IEEE 754 single (32-bit) → half (16-bit). Round-to-nearest, no
     *  subnormal support — adequate for SD UNet inputs where values
     *  sit comfortably within fp16's normal range (~-65504..65504,
     *  smallest normal ~6e-5). Used to feed fp16 OnnxTensor inputs. */
    private fun floatToHalfBits(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        var expF = (bits ushr 23) and 0xff
        var mantF = bits and 0x7fffff
        if (expF == 255) {
            // NaN / Infinity
            return (sign or 0x7c00 or (if (mantF != 0) 0x200 else 0)).toShort()
        }
        val newExp = expF - 127 + 15
        return when {
            newExp <= 0 -> sign.toShort()                    // underflow → ±0
            newExp >= 0x1f -> (sign or 0x7c00).toShort()    // overflow → ±Inf
            else -> {
                // round mantissa: drop bottom 13 bits with round-to-nearest-even
                val rounded = mantF + 0x1000
                val mant16 = (rounded ushr 13) and 0x3ff
                // if the round bumped mantissa overflow, increment exponent
                val carry = (rounded ushr 23) and 0x1
                val finalExp = newExp + carry
                if (finalExp >= 0x1f) (sign or 0x7c00).toShort()
                else (sign or (finalExp shl 10) or mant16).toShort()
            }
        }
    }

    /** half (16-bit) → IEEE 754 single (32-bit). */
    private fun halfBitsToFloat(h: Short): Float {
        val bits = h.toInt() and 0xffff
        val sign = (bits and 0x8000) shl 16
        val exp = (bits and 0x7c00) ushr 10
        val mant = bits and 0x3ff
        return when (exp) {
            0 -> {
                if (mant == 0) java.lang.Float.intBitsToFloat(sign)
                else {
                    // subnormal — convert to fp32 normalized
                    var m = mant
                    var e = -14
                    while ((m and 0x400) == 0) {
                        m = m shl 1
                        e -= 1
                    }
                    val mantF = (m and 0x3ff) shl 13
                    val expF = (e + 127) shl 23
                    java.lang.Float.intBitsToFloat(sign or expF or mantF)
                }
            }
            0x1f -> {
                // Inf / NaN
                java.lang.Float.intBitsToFloat(sign or 0x7f800000 or (mant shl 13))
            }
            else -> {
                val expF = (exp - 15 + 127) shl 23
                val mantF = mant shl 13
                java.lang.Float.intBitsToFloat(sign or expF or mantF)
            }
        }
    }

    /** Wrap a FloatArray as a Direct ByteBuffer of fp16 values. The
     *  ByteBuffer is what `OnnxTensor.createTensor(env, buf, shape,
     *  OnnxJavaType.FLOAT16)` expects; size = arr.size × 2 bytes. */
    private fun floatArrayToHalfBuffer(arr: FloatArray): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(arr.size * 2).order(ByteOrder.nativeOrder())
        for (v in arr) buf.putShort(floatToHalfBits(v))
        buf.rewind()
        return buf
    }

    /** Read a FLOAT16 OnnxTensor's payload into a FloatArray of length
     *  `count`. Tensor's underlying buffer is unsigned shorts; we go
     *  through a ShortBuffer and convert one element at a time. */
    private fun halfTensorToFloatArray(tensor: OnnxTensor, count: Int): FloatArray {
        val raw = tensor.byteBuffer
        raw.order(ByteOrder.nativeOrder())
        val sb = raw.asShortBuffer()
        val out = FloatArray(count)
        for (i in 0 until count) out[i] = halfBitsToFloat(sb.get(i))
        return out
    }

    /** SessionOptions tuned for lower peak RSS — at the cost of a small
     *  speed hit. Pattern optimization preallocates intermediate tensors
     *  and the CPU arena hangs onto big slabs across sessions; both make
     *  Android's lowmemorykiller more likely to evict us mid-inference
     *  when the device already has lots of background apps resident. */
    private fun sessionOptions(): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        opts.setMemoryPatternOptimization(false)
        opts.setCPUArenaAllocator(false)
        return opts
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
            val sampleShape = longArrayOf(2, 4, latentH.toLong(), latentW.toLong())
            val hiddenShape = longArrayOf(2, 77, 768)
            val tsShape     = longArrayOf(2)

            if (unetIsFp16) {
                sampleT = OnnxTensor.createTensor(
                    ortEnv, floatArrayToHalfBuffer(sampleBatch),
                    sampleShape, OnnxJavaType.FLOAT16,
                )
                tsT = OnnxTensor.createTensor(
                    ortEnv, floatArrayToHalfBuffer(tsBatch),
                    tsShape, OnnxJavaType.FLOAT16,
                )
                hiddenT = OnnxTensor.createTensor(
                    ortEnv, floatArrayToHalfBuffer(hiddenBatch),
                    hiddenShape, OnnxJavaType.FLOAT16,
                )
            } else {
                sampleT = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(sampleBatch), sampleShape,
                )
                tsT = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(tsBatch), tsShape,
                )
                hiddenT = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(hiddenBatch), hiddenShape,
                )
            }

            val out = session.run(mapOf(
                "sample"               to sampleT,
                "timestep"             to tsT,
                "encoder_hidden_states" to hiddenT,
            ))
            val outSample = out.get("out_sample").get() as OnnxTensor
            val flat = if (unetIsFp16) {
                halfTensorToFloatArray(outSample, 2 * perFrame)
            } else {
                FloatArray(2 * perFrame).also { outSample.floatBuffer.get(it) }
            }
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
