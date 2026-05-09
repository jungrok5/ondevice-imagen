package com.jungrok5.drawmoment

import android.content.Intent
import org.godotengine.godot.Godot
import org.godotengine.godot.plugin.GodotPlugin
import org.godotengine.godot.plugin.SignalInfo
import org.godotengine.godot.plugin.UsedByGodot

/**
 * Native Android backing for the "Draw this moment" workflow.
 *
 * GDScript side:
 *   - calls start_inference(prompt, output_path) — kicks off a
 *     Foreground Service that does the long inference work.
 *   - listens for `inference_completed(output_path)` and
 *     `inference_failed(error_msg)` signals.
 *
 * Why a Foreground Service and not just a WorkManager Worker?
 * - SD 1.5 INT8 inference on a Note 10+ takes 1–3 min. SDXL-Turbo can
 *   take 5–15 min. WorkManager doesn't guarantee continuous foreground
 *   priority for that long; the OS may pause it. Foreground Service
 *   with ongoing notification = continuous CPU/GPU access.
 * - The Service posts an "in progress" notification while running, then
 *   replaces it with a "Done — tap to view" notification on success.
 *
 * Phase 1: the Service body is a 30 s sleep + write a placeholder PNG.
 * Phase 2: replace the body with ONNX Runtime SD 1.5 Session.run().
 */
class DrawMomentPlugin(godot: Godot) : GodotPlugin(godot) {

    override fun getPluginName(): String = "DrawMomentPlugin"

    override fun onGodotSetupCompleted() {
        super.onGodotSetupCompleted()
        liveInstance = this
    }

    override fun onMainDestroy() {
        super.onMainDestroy()
        if (liveInstance === this) {
            liveInstance = null
        }
    }

    override fun getPluginSignals(): MutableSet<SignalInfo> = mutableSetOf(
        SignalInfo("inference_completed", String::class.java),
        SignalInfo("inference_failed",    String::class.java),
    )

    @UsedByGodot
    fun start_inference(prompt: String, outputPath: String) {
        val ctx = godot.getActivity()?.applicationContext ?: return
        val intent = Intent(ctx, InferenceForegroundService::class.java).apply {
            putExtra(InferenceForegroundService.EXTRA_PROMPT, prompt)
            putExtra(InferenceForegroundService.EXTRA_OUTPUT_PATH, outputPath)
        }
        // startForegroundService is required on Android 8+ — the Service
        // then calls startForeground() within ~5 s or the OS kills it.
        ctx.startForegroundService(intent)
    }

    /** Called by the Service when inference finishes. Forwards to GDScript. */
    fun emitCompleted(outputPath: String) {
        emitSignal("inference_completed", outputPath)
    }

    /** Called by the Service when inference fails. Forwards to GDScript. */
    fun emitFailed(errorMsg: String) {
        emitSignal("inference_failed", errorMsg)
    }

    companion object {
        // Service writes here so it can fire signals on the live plugin
        // instance regardless of which Activity bounces it.
        @Volatile
        var liveInstance: DrawMomentPlugin? = null
    }
}
