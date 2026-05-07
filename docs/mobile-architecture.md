# On-device architecture (Android / iOS)

This is the target architecture once the PC prototype validates the aesthetic.

## High-level flow

```
[Sensors / APIs]                [On-device storage]
  - Water log (manual)             SQLite / Room / CoreData
  - Weather (system + API cache) ->  daily entries, encrypted at rest
  - GPS POI (FusedLocation /
    CoreLocation + reverse geocode)
                                       |
                                       v
                          [End-of-week aggregator]
                                       |
                                       v
                           [Prompt builder (rule-based)]
                                       |
                                       v
                          [Base collage renderer (Skia)]
                                       |
                                       v
                       [Stable Diffusion img2img on-device]
                                       |
                                       v
                       [Postcard composer + share / save]
```

Everything except the optional weather lookup is fully on-device, matching the
"private journal" intent.

## Image generation per platform

### iOS — Apple Core ML Stable Diffusion
- Repo: `apple/ml-stable-diffusion` (official, MIT)
- Convert SD 1.5 (or SD-Turbo) to Core ML packages once, ship inside the app
  bundle or download on first launch.
- 6-bit palettization brings SD 1.5 weights to ~700 MB.
- Performance budget on iPhone 14: ~3-6 s for 4 steps at 512x512 with the
  Neural Engine.
- API surface used: `StableDiffusionPipeline` (Swift), with `Image2Image`
  variant for img2img.
- Requires iOS 16.2+ for ANE-friendly ops; iOS 17+ recommended.

### Android — Google AI Edge / MediaPipe Image Generator
- The MediaPipe Tasks `ImageGenerator` ships an LCM-distilled SD 1.5 with a
  pre-baked schedule. Single `.task` bundle, ~1.5 GB.
- Device floor: Snapdragon 8 Gen 2 / Tensor G3 with at least 6 GB RAM. Older
  devices fall back to a "no doodle today" message rather than crashing.
- API: `ImageGenerator` + `ImageGeneratorOptions` (Kotlin). Conditioning via
  text prompt; img2img is exposed through the `ConditionOptions.imageConditionOptions`
  (edge / depth / face) — for our use case, **edge conditioning** on the
  Skia-rendered base collage gives the closest analogue to img2img on iOS.
- Alternative: ONNX Runtime Mobile + a quantized SD 1.5 ONNX export, with
  NNAPI / QNN delegates. More work, more control.

### Cross-platform shortcut
If you can tolerate one inference path: use **ONNX Runtime Mobile** with a
quantized SD 1.5 ONNX export from `optimum`. Same model file on both
platforms, with NNAPI on Android and Core ML EP on iOS. Loses some of the
ANE optimization Apple provides but reduces engineering surface to one
pipeline instead of two.

## Why SD 1.5 (not SDXL / SD3)

- Aesthetic target is "intentionally bad" — model fidelity is *not* the
  bottleneck. SD 1.5's lower sharpness even helps.
- Mobile RAM/disk budgets favor SD 1.5: weights fit under ~1.5 GB after
  quantization vs ~5+ GB for SDXL.
- LCM / Turbo distillations reduce step count to 1-4, making generation
  feel interactive.

## Data layer

- **Android**: Room + EncryptedSharedPreferences for the encryption key.
- **iOS**: CoreData with NSPersistentCloudKitContainer disabled (keep local).
- **Schema**: one row per (water_event | weather_snapshot | place_visit),
  plus a `weeks` table caching aggregated summaries.
- **Privacy**: GPS coordinates reverse-geocoded to a POI name on-device when
  possible, then the raw lat/lng is dropped before the row is committed.

## Background scheduling (the actual user flow)

The user only wants a postcard once a week, so the generation job runs as
opportunistic background work — no "tap and wait" UX. Both platforms have
APIs designed for exactly this shape of work, which means slow inference is
*fine*: the device is plugged in, locked, and idle anyway.

### iOS — `BGProcessingTask`

```swift
let req = BGProcessingTaskRequest(identifier: "ai.localrnd.weeklyPostcard")
req.requiresExternalPower       = true   // only while charging
req.requiresNetworkConnectivity = false  // fully offline
req.earliestBeginDate           = nextSundayMorning
BGTaskScheduler.shared.submit(req)
```

- iOS 13+. The system schedules opportunistically — typically while
  charging and locked. No exact time guarantee.
- Multi-minute jobs are explicitly supported (it is the documented use
  case for ML model training and photo-library indexing).
- ANE-backed SD 1.5 finishes in ~30-60 s — well within any background
  window, with battery headroom to spare.
- User-side gotcha: if Background App Refresh is disabled for the app,
  `BGProcessingTask` never fires. UI must communicate this gracefully.

### Android — WorkManager + Foreground Service

```kotlin
val req = OneTimeWorkRequestBuilder<PostcardWorker>()
    .setConstraints(Constraints.Builder()
        .setRequiresCharging(true)
        .setRequiresBatteryNotLow(true)
        .setRequiresDeviceIdle(true)
        .build())
    .setInitialDelay(7, TimeUnit.DAYS)
    .build()
WorkManager.getInstance(ctx).enqueue(req)
```

- WorkManager negotiates Doze and battery-optimization policy for us.
- Inference longer than ~5 min must be promoted to a Foreground Service.
  Android 15 added a `mediaProcessing` foreground-service type fitting
  exactly this case; pre-15, use `dataSync`.
- Notification is mandatory while the FG service is running ("Drawing
  this week's postcard…").
- Even on devices without NPU, a CPU fallback of several minutes is fine
  here — the constraint stack guarantees we are charging and idle.

### Implication for inference budget

Because the work runs in the background under charging + idle, the
acceptable inference time is *minutes*, not seconds. That widens the
choice of models meaningfully:

- iPhone 14+ ANE → SDXL 1024² is feasible at ~30-60 s.
- Snapdragon 8 Gen 2 NPU → MediaPipe LCM-SD1.5 at ~10 s.
- Older devices (CPU fallback) → SD 1.5 at 4-8 min — still acceptable.

UX flow: user opens the app on Sunday, finds last week's postcard already
generated. No spinner, no wait — the work happened while the phone was
charging on the nightstand.

## Open questions to validate via the PC prototype

1. Does the prompt actually produce the desired aesthetic, or do we need a
   "child-doodle" LoRA layered on top?
2. Is img2img (collage seed) necessary, or is txt2img with the prompt alone
   good enough? img2img is more expensive on mobile.
3. How many inference steps are the minimum that still look intentional (vs.
   noisy garbage)?
4. What seed-stability do we want? Fixing the seed per user gives them a
   coherent visual diary; randomizing makes each week feel fresh.
