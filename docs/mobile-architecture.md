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

## The "no-conversion" path (recommended starting plan)

Before considering custom conversion / LoRA-merge work, check if the
checkpoint we want already ships pre-converted. For most popular SD 1.5
fine-tunes the answer is yes:

- **iOS**: the `coreml-community` org on Hugging Face hosts **~145
  pre-converted `.mlpackage` artifacts**. Notably, the `Dreamshaper-8`
  checkpoint we used for the watercolor result is already there as
  `coreml-community/coreml-DreamShaper-v8_cn`. Drop into the app,
  load with Apple's `StableDiffusionPipeline` Swift library, done — no
  conversion code, no Xcode toolchain pain on our side.
- **Android**: there is no equivalent curated bundle of pre-converted
  community fine-tunes. The two viable paths are:
  - `optimum-cli export onnx --model Lykon/dreamshaper-8 ...` — one
    command, ~10 min, produces an ONNX bundle that ONNX Runtime Mobile
    runs directly on both NNAPI and Qualcomm QNN.
  - MediaPipe `ImageGenerator` with whatever Google ships — fast and
    well-supported, but limited to their bundled LCM-SD1.5; the
    watercolor aesthetic may lose some fidelity.

### LoRA is also not strictly required

The watercolor sample committed to `samples/style_watercolor_diary.png`
was produced with **only the prompt** — no LoRA. This matters for the
mobile plan because it means style is defined in a string, not a binary
adapter, so the entire "LoRA hot-swap" problem disappears for the
default aesthetic. Style variants (riso, crayon, ghibli, minhwa) that
the base model does *not* know are the ones that need LoRAs and the
per-style merged-checkpoint shipping pattern. That is Phase 2 work.

### Concrete recommended stack

```
Aesthetic        : watercolor diary (prompt-only)
Base checkpoint  : Lykon/dreamshaper-8
iOS              : coreml-community/coreml-DreamShaper-v8_cn
                   + apple/ml-stable-diffusion Swift package
                   (~700MB-1GB on disk after 6-bit palettization)
Android          : ONNX export of the same checkpoint via optimum-cli
                   + ONNX Runtime Mobile (NNAPI/QNN delegates)
                   (~1.5GB on disk)
Prompt           : src/prompt_builder.py output, fed straight to either
                   pipeline — no per-platform divergence
Step count       : 4 (LCM/Turbo) for foreground; 30 (Euler-a) for
                   background-job mode while charging + idle
```

This stack keeps "specific checkpoint" and "specific style" out of the
conversion / engineering loop until we have a reason to add a second
aesthetic.

## What is verified vs. assumed (as of this commit)

The PC prototype proves the *aesthetic + pipeline shape*. It does NOT prove
that any specific checkpoint we are using will actually run on iOS or
Android — that is a downstream engineering task. Be precise:

| Tested on this PC | Mobile-portable as-is? |
|---|---|
| `stabilityai/sd-turbo` (doodle path) | ✅ Both Apple ml-stable-diffusion and MediaPipe ImageGenerator ship matching SD-Turbo / LCM-distilled SD 1.5 templates. |
| `Lykon/dreamshaper-8` (watercolor, realistic) | ⚠️ Architecture-compatible but requires **per-checkpoint conversion**: Core ML `.mlpackage` for iOS (~700 MB at 6-bit), ONNX export for Android (MediaPipe does not accept arbitrary fine-tunes — must drop down to ONNX Runtime Mobile). |
| `dreamlike-art/dreamlike-anime-1.0` (anime) | ⚠️ Same as above — every fine-tune is a separate conversion + ship/download artifact. |
| LoRA-stacked styles (riso, crayon, ink wash) | ❌ Neither Core ML nor MediaPipe supports LoRA hot-swap. Each LoRA must be **merged into the base checkpoint** before conversion, producing one shipped model per style. |
| 30-step Euler-a inference | △ Architecture-fine, but on mobile you usually want LCM/Turbo (4 steps) for power. Background-job mode (charging + idle) makes 30 steps acceptable too — see Background scheduling above. |

### Implications for app size

Each style = one converted model artifact:
- Quantized SD 1.5 (Core ML 6-bit): ~700 MB - 1 GB
- ONNX-exported SD 1.5: ~1.5-2 GB

A "5 styles bundled" app would be 4-10 GB — unrealistic. Realistic plan:

1. **Bundle one default style** (the user's primary aesthetic) inside the
   app binary.
2. **Other styles are optional downloads**, fetched on first selection from
   a CDN. Users pick their look once, download once, regenerate weekly
   forever after.

### What is NOT validated yet

- Actually running any of these checkpoints on a real iPhone or Android
  phone. The Core ML / MediaPipe paths are documented and well-trodden,
  but our specific checkpoints have not been exported, profiled, or
  benchmarked on device.
- LoRA-merge → checkpoint → convert pipelines (paper-tested only).
- Whether the watercolor aesthetic survives the 6-bit Core ML
  quantization. It probably does (the look is forgiving), but should be
  verified before locking the visual direction.

These are the natural Phase 2 tasks: pick the visual direction here on
the PC, then prove the iOS path end-to-end with the chosen checkpoint
before doing Android.

## Open questions to validate via the PC prototype

1. Does the prompt actually produce the desired aesthetic, or do we need a
   "child-doodle" LoRA layered on top?
2. Is img2img (collage seed) necessary, or is txt2img with the prompt alone
   good enough? img2img is more expensive on mobile.
3. How many inference steps are the minimum that still look intentional (vs.
   noisy garbage)?
4. What seed-stability do we want? Fixing the seed per user gives them a
   coherent visual diary; randomizing makes each week feel fresh.
