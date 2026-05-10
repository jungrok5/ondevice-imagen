package com.jungrok5.drawmoment

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
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

    /** Permission dialog result lands here on the main thread.
     *  If the user finally grants location, kick off a fresh fix so
     *  the UI doesn't sit on the "no permission" status forever. */
    override fun onMainRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onMainRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != LOCATION_PERMISSION_REQ) return
        val granted = grantResults.any { it == PackageManager.PERMISSION_GRANTED }
        if (granted) {
            request_current_location()
        } else {
            emitSignal("gps_failed", "permission denied")
        }
    }

    override fun getPluginSignals(): MutableSet<SignalInfo> = mutableSetOf(
        SignalInfo("inference_completed", String::class.java),
        SignalInfo("inference_failed",    String::class.java),
        // gps_updated payload is "lat,lon" so we don't have to fight
        // Godot's signal-arg numeric-type plumbing.
        SignalInfo("gps_updated",         String::class.java),
        SignalInfo("gps_failed",          String::class.java),
    )

    @UsedByGodot
    fun start_inference(prompt: String, outputPath: String) {
        val ctx = godot.getActivity()?.applicationContext ?: return
        // Phase 3: route through WorkManager so the same code path can
        // serve both the user-tap dev flow (OneTimeWorkRequest) and the
        // weekly-postcard schedule (PeriodicWorkRequest, 3-3 next).
        // KEEP policy makes a second tap during a running job a no-op
        // instead of queueing — matches user expectation that the button
        // dims while inference is in flight.
        val data = Data.Builder()
            .putString(PostcardWorker.KEY_PROMPT, prompt)
            .putString(PostcardWorker.KEY_OUTPUT_PATH, outputPath)
            .build()
        val req = OneTimeWorkRequestBuilder<PostcardWorker>()
            .setInputData(data)
            .build()
        WorkManager.getInstance(ctx).enqueueUniqueWork(
            PostcardWorker.UNIQUE_NAME,
            ExistingWorkPolicy.KEEP,
            req,
        )
    }

    /** Called by the Service when inference finishes. Forwards to GDScript. */
    fun emitCompleted(outputPath: String) {
        emitSignal("inference_completed", outputPath)
    }

    /** Called by the Service when inference fails. Forwards to GDScript. */
    fun emitFailed(errorMsg: String) {
        emitSignal("inference_failed", errorMsg)
    }

    // --- GPS --------------------------------------------------------------

    /** True if the user has already granted FINE or COARSE location. */
    @UsedByGodot
    fun has_location_permission(): Boolean {
        val act = godot.getActivity() ?: return false
        return ContextCompat.checkSelfPermission(act, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED ||
               ContextCompat.checkSelfPermission(act, Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
    }

    /** Pops the system location-permission dialog if needed. The result
     *  lands in onMainRequestPermissionsResult, but for v1 we don't
     *  surface that callback — the next has_location_permission() call
     *  is the source of truth. */
    @UsedByGodot
    fun request_location_permission() {
        val act = godot.getActivity() ?: return
        val needs = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(act, Manifest.permission.ACCESS_FINE_LOCATION) !=
                PackageManager.PERMISSION_GRANTED) {
            needs.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        if (ContextCompat.checkSelfPermission(act, Manifest.permission.ACCESS_COARSE_LOCATION) !=
                PackageManager.PERMISSION_GRANTED) {
            needs.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        if (needs.isNotEmpty()) {
            ActivityCompat.requestPermissions(act, needs.toTypedArray(), LOCATION_PERMISSION_REQ)
        }
    }

    /** Asks for a single fresh location fix. On Android 11+ we use
     *  getCurrentLocation; below that we fall back to last-known.
     *  Result arrives via the gps_updated / gps_failed signals so
     *  GDScript can stay async. */
    @UsedByGodot
    fun request_current_location() {
        Log.d(TAG, "request_current_location: entry")
        val act = godot.getActivity() ?: run {
            Log.w(TAG, "request_current_location: no activity")
            emitSignal("gps_failed", "no activity")
            return
        }
        if (!has_location_permission()) {
            Log.w(TAG, "request_current_location: no permission")
            emitSignal("gps_failed", "no permission")
            return
        }
        val lm = act.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: run {
            Log.w(TAG, "request_current_location: no LocationManager")
            emitSignal("gps_failed", "no LocationManager")
            return
        }
        // Try GPS, NETWORK, then PASSIVE — collect fixes from all enabled
        // providers and surface whichever returned anything first. Mobile
        // location services are often half-on (NETWORK enabled, GPS off
        // because the user is indoors).
        val gpsOn = lm.isProviderEnabled(LocationManager.GPS_PROVIDER)
        val netOn = lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
        Log.d(TAG, "request_current_location: providers gps=$gpsOn net=$netOn sdk=${Build.VERSION.SDK_INT}")
        val provider = when {
            gpsOn -> LocationManager.GPS_PROVIDER
            netOn -> LocationManager.NETWORK_PROVIDER
            else -> {
                Log.w(TAG, "request_current_location: no provider enabled")
                emitSignal("gps_failed", "no provider enabled — turn on Location in Settings")
                return
            }
        }
        try {
            // Hand back the cached fix immediately so the UI can update
            // even before a fresh fix arrives — useful on cold boot when
            // GPS_PROVIDER may take 30+ s to converge. Also pull NETWORK's
            // last-known if GPS doesn't have one.
            var cached = lm.getLastKnownLocation(provider)
            if (cached == null && provider == LocationManager.GPS_PROVIDER && netOn) {
                cached = lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            }
            if (cached != null) {
                Log.d(TAG, "request_current_location: cached fix ${cached.latitude},${cached.longitude} from ${cached.provider}")
                emitSignal("gps_updated", "${cached.latitude},${cached.longitude}")
            } else {
                Log.d(TAG, "request_current_location: no cached fix")
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                lm.getCurrentLocation(
                    provider,
                    null,
                    ContextCompat.getMainExecutor(act),
                ) { loc ->
                    if (loc != null) {
                        Log.d(TAG, "getCurrentLocation: fresh ${loc.latitude},${loc.longitude}")
                        emitSignal("gps_updated", "${loc.latitude},${loc.longitude}")
                    } else if (cached == null) {
                        Log.w(TAG, "getCurrentLocation: returned null, no cached fallback")
                        emitSignal("gps_failed", "fix unavailable (move outdoors or wait)")
                    }
                }
            } else if (cached == null) {
                emitSignal("gps_failed", "no last-known fix on pre-R device")
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "request_current_location: SecurityException", e)
            emitSignal("gps_failed", "security: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "DrawMomentPlugin"
        private const val LOCATION_PERMISSION_REQ = 4711

        // Service writes here so it can fire signals on the live plugin
        // instance regardless of which Activity bounces it.
        @Volatile
        var liveInstance: DrawMomentPlugin? = null
    }
}
