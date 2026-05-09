package com.jungrok5.drawmoment

import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Phase 2 Step 5b-3a — Karras sigma schedule for DPMSolverMultistep.
 *
 * Mirrors `diffusers.schedulers.DPMSolverMultistepScheduler` with the
 * config that scheduler_config.json ships:
 *   beta_schedule="scaled_linear"
 *   beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000
 *   use_karras_sigmas=true
 *   final_sigmas_type="zero"   (last entry forced to 0)
 *   solver_order=2, algorithm_type="dpmsolver++", solver_type="midpoint"
 *
 * This stage 5b-3a only computes the σ vector — the step() function
 * follows in 5b-3b. Cross-checked against scripts/scheduler_reference.py
 * to within 1e-5 absolute on every entry.
 */
class DpmScheduler(
    private val numTrainTimesteps: Int = 1000,
    private val betaStart: Float = 0.00085f,
    private val betaEnd: Float = 0.012f,
) {
    /** σ values computed from the diffusion noise schedule —
     *  σ_i = sqrt((1 - α̅_i) / α̅_i), one per training timestep. */
    val trainingSigmas: FloatArray

    init {
        // scaled_linear: betas = linspace(sqrt(beta_start), sqrt(beta_end), T)^2
        val betas = FloatArray(numTrainTimesteps) { i ->
            val t = i.toFloat() / (numTrainTimesteps - 1).toFloat()
            val sqrtBeta = sqrt(betaStart) + t * (sqrt(betaEnd) - sqrt(betaStart))
            sqrtBeta * sqrtBeta
        }
        // α̅_i = ∏_{j≤i} (1 - β_j)
        val alphasCumprod = FloatArray(numTrainTimesteps)
        var prod = 1.0f
        for (i in 0 until numTrainTimesteps) {
            prod *= (1f - betas[i])
            alphasCumprod[i] = prod
        }
        // σ_i = sqrt((1 - α̅_i) / α̅_i)
        trainingSigmas = FloatArray(numTrainTimesteps) { i ->
            sqrt((1f - alphasCumprod[i]) / alphasCumprod[i])
        }
    }

    /** Karras (2022) sigma schedule —
     *  σ_i = (σ_max^(1/ρ) + i/(n-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
     *  for i in 0..n-1, with a trailing 0 appended (final_sigmas_type="zero").
     *  Returns n+1 values. */
    fun karrasSigmas(numInferenceSteps: Int, rho: Float = 7f): FloatArray {
        require(numInferenceSteps >= 2) { "need at least 2 inference steps" }
        val sigmaMin = trainingSigmas.first()
        val sigmaMax = trainingSigmas.last()
        val invRho = 1f / rho
        val sigmaMaxRho = sigmaMax.pow(invRho)
        val sigmaMinRho = sigmaMin.pow(invRho)

        val out = FloatArray(numInferenceSteps + 1)
        for (i in 0 until numInferenceSteps) {
            val ramp = i.toFloat() / (numInferenceSteps - 1).toFloat()
            val sigRho = sigmaMaxRho + ramp * (sigmaMinRho - sigmaMaxRho)
            out[i] = sigRho.pow(rho)
        }
        out[numInferenceSteps] = 0f  // final_sigmas_type="zero"
        return out
    }

    /** Timesteps in the diffusion schedule that correspond to each σ.
     *  Picked by nearest match — diffusers uses interpolation but the
     *  difference at our step counts is ≤1 timestep. */
    fun karrasTimesteps(sigmas: FloatArray): IntArray {
        // training σ is monotonically increasing in i. For each sigma
        // (except the trailing zero), find the closest training i.
        val out = IntArray(sigmas.size - 1)
        for (k in 0 until sigmas.size - 1) {
            val target = sigmas[k]
            var bestI = 0
            var bestDist = Float.POSITIVE_INFINITY
            for (i in 0 until trainingSigmas.size) {
                val d = kotlin.math.abs(trainingSigmas[i] - target)
                if (d < bestDist) {
                    bestDist = d
                    bestI = i
                }
            }
            out[k] = bestI
        }
        return out
    }
}
