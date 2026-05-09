# android-plugin

Native Android module that backs the Godot client's "Draw this moment"
button. Builds to a single `.aar` that the Godot Android export bundles.

## Phase 1 — workflow only (no real inference)

What works after phase 1:

1. GDScript calls `DrawMomentPlugin.start_inference(prompt, output_path)`.
2. `InferenceForegroundService` shows an ongoing "Drawing this moment..."
   notification and runs a 30 s placeholder job that writes a tiny PNG.
3. On completion, it posts a "Done — tap to view" notification with a
   PendingIntent that launches the app.
4. GDScript receives `inference_completed(output_path)` and loads the PNG.

What this verifies (the *real* point of phase 1):

- Foreground Service stays alive long enough on Android 12/13/14
- Notification permission flow works (POST_NOTIFICATIONS on 13+)
- PendingIntent re-launches Godot back into the result view
- GDScript ↔ Kotlin signal round-trip works

## Phase 2 — replace the placeholder with ONNX Runtime SD 1.5

Single point of change: `InferenceForegroundService.doInference()`.

Add the dep in `draw_moment_plugin/build.gradle`:

```gradle
implementation "com.microsoft.onnxruntime:onnxruntime-android:1.18.0"
```

Bundle the ONNX model assets under `client/android/build/assets/models/`
(or load from `user://` after a one-time download — model weights are
~600 MB so probably first-launch download with progress UI is friendlier
than fattening the APK).

## Build

Plain Gradle wrapper. Note 10+ supports minSdk 24, so we don't need
anything fancy:

```powershell
# From this directory
.\gradlew assembleRelease
# Output: draw_moment_plugin/build/outputs/aar/draw_moment_plugin-release.aar
```

Drop the .aar (and a matching `*.gdap` config) under
`client/android/plugins/` so Godot's Android export picks it up.

The Gradle wrapper itself (`gradlew`, `gradlew.bat`, `gradle/wrapper/`)
isn't checked in yet — `gradle init --type basic` from this directory
generates them on first setup, or copy from a sibling project (e.g.
WaterBloom).

## Permissions

Requested in `AndroidManifest.xml`:

| Permission | Why |
|---|---|
| `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_DATA_SYNC` | Long inference job (1–15 min) |
| `POST_NOTIFICATIONS` | Android 13+ runtime permission for the progress + done notifications |
| `INTERNET` | Open-Meteo + Nominatim from Godot's HTTPRequest |
| `ACCESS_COARSE_LOCATION` / `ACCESS_FINE_LOCATION` | Phase 2 GPS pull (currently lat/lon LineEdit) |

The runtime-permission dialogs are triggered from GDScript before the
first inference; the plugin currently assumes they've been granted.
