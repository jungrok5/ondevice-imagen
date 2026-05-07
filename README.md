# local-ai-rnd

Research notes on what on-device image generation can do in 2026 — what
models run on iPhone and Android phones without per-platform engineering,
how good the output gets, and how the speed / quality / aesthetic dials
trade off.

The PC scaffold here is the testbed: a small diffusers + Gradio harness
that lets us try a checkpoint, look at the result, and decide whether it
is worth carrying to mobile. The committed `samples/` images and the
notes below are the actual research output.

## Hardware envelope of this PC

- AMD Radeon Pro 580X (4 GB), CPU torch only — no CUDA, no torch-directml
- Python 3.9, diffusers + transformers + Gradio
- Reference numbers below are for **CPU FP32 inference**. For comparison:
  - DirectML on the Radeon: ~5-10× faster than these numbers
  - iPhone 14+ Apple Neural Engine: ~3-10 s for the same workloads
  - Snapdragon 8 Gen 2 Hexagon NPU: ~5-10 s

That gap matters because everything below is targeting "what runs on a
phone in the background while charging" — slow on PC ≠ slow on device.

## Speed dial: SD-Turbo (1-4 steps)

SD-Turbo is the "fast" end. Single-step generation, ~1.4 GB model.

| input | output |
|---|---|
| Rule-rendered pictographic collage from `data/sample_week.json` (used as the img2img seed) | ![base](samples/base_collage.png) |
| img2img redraw — 32.5 s on CPU, `strength=0.99`, 4 steps | ![img2img](samples/sd_img2img.png) |
| txt2img — 24.3 s on CPU, 4 steps | ![txt2img](samples/sd_txt2img.png) |

Three lessons (see commits `0922e01` and `ec7dc20`):

1. **Front-load style tags.** CLIP truncates at 77 tokens; whatever is at
   the front of the prompt dominates. Putting style at the tail silently
   loses the style.
2. **Keep text off the img2img seed.** SD interprets text in the seed as
   text and emits garbled fake text in the output — use pure pictograms.
3. **`strength * steps` is the denoising budget.** On 4-step SD-Turbo,
   `strength < 0.95` leaves the seed image too visible.

## Quality dial: SD 1.5 fine-tunes (30 steps Euler-a)

The opposite end. Same prompt + seed across both:

| model | runtime (CPU) | output |
|---|---|---|
| `Lykon/dreamshaper-8` (realistic / versatile) | 246 s | ![realistic](samples/quality_realistic.png) |
| `dreamlike-art/dreamlike-anime-1.0` (anime) | 249 s | ![anime](samples/quality_anime.png) |

So the PC ceiling is "magazine-grade SD 1.5 in ~4 min/image" on plain CPU
torch. The same workload finishes in 3-10 s on a current-gen phone NPU,
which is why on-device generation is finally viable in 2026.

## Aesthetic dial: prompt-only style on a single base model

`Lykon/dreamshaper-8` + style prefix in the prompt, same subject + seed
across all entries — the only varying axis is the aesthetic preset
([src/styles.py](src/styles.py)).

| preset | output | finding |
|---|---|---|
| `watercolor_diary` | ![watercolor](samples/style_watercolor_diary.png) | **lands cleanly** — paper texture, pastel washes, journal composition all read as watercolor |
| `risograph_zine` | ![riso](samples/style_risograph_zine.png) | palette landed but halftone / grain / registration drift missing → just a blue-tinted photo. Needs a riso LoRA. |
| `crayon_picturebook` | ![crayon](samples/style_crayon_picturebook.png) | warm composition but lacks crayon physicality → reads as digital illustration. Also needs a LoRA. |

**Generalization**: SD 1.5 fine-tunes are strong on subject and lighting,
weak on medium-specific *physicality*. Anything where the texture of the
medium is the point (riso, crayon, linocut) needs a dedicated LoRA. Looks
like watercolor or painterly illustration come essentially for free.

## Cross-platform availability: which models ship without our own conversion

The interesting practical question for on-device deployment is: which
checkpoints are already pre-converted for Apple Core ML (so iOS is a
zero-conversion drop-in) AND can be ONNX-exported for Android via a
single `optimum-cli` command?

| base checkpoint | iOS pre-converted (Hugging Face) | Android (ONNX) |
|---|---|---|
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `apple/coreml-stable-diffusion-v1-5` (official Apple) | `optimum-cli export onnx` |
| `stabilityai/sd-turbo` | `apple/coreml-*` (official Apple) | `optimum-cli export onnx` |
| `Lykon/dreamshaper-8` | `coreml-community/coreml-DreamShaper-v8_cn` | `optimum-cli export onnx` |
| `dreamlike-art/dreamlike-diffusion-1.0` | `coreml-community/coreml-dreamlike-diffusion` | `optimum-cli export onnx` |
| `dreamlike-art/dreamlike-photoreal-2.0` | covered by `coreml-community` | `optimum-cli export onnx` |
| `dreamlike-art/dreamlike-anime-1.0` | covered by `coreml-community` | `optimum-cli export onnx` |
| `prompthero/openjourney` | covered by `coreml-community` | `optimum-cli export onnx` |
| `SG161222/Realistic_Vision_V5.1_noVAE` | `coreml-community/coreml-realisticVision-v51VAE_cn` | `optimum-cli export onnx` |

Reference samples generated on this PC for the four models we hadn't
already exercised (same prompt + seed across all four):

| model | output |
|---|---|
| `stable-diffusion-v1-5` (baseline) | ![sd15](samples/cross_sd_1_5_base.png) |
| `dreamlike-diffusion-1.0` | ![ddiff](samples/cross_dreamlike_diffusion.png) |
| `openjourney` | ![oj](samples/cross_openjourney.png) |
| `dreamlike-photoreal-2.0` | ![dphoto](samples/cross_dreamlike_photoreal_2.png) |

(See [scripts/cross_platform_demo.py](scripts/cross_platform_demo.py).)

## Mobile architecture notes

The full notes on background-task scheduling, Core ML / MediaPipe
deployment, app size, and what is *verified vs. assumed* live in
[docs/mobile-architecture.md](docs/mobile-architecture.md). Highlights:

- **iOS**: pre-converted `.mlpackage` via Apple's `ml-stable-diffusion`
  Swift library. No conversion code needed for any of the models above.
- **Android**: `optimum-cli export onnx` once → ONNX Runtime Mobile with
  NNAPI / QNN delegates. Same checkpoint, same prompt, same result.
- **Background scheduling** (`BGProcessingTask` on iOS, `WorkManager` +
  Foreground Service on Android) makes the "slow inference is fine" path
  viable: phone runs the model overnight while charging.
- **LoRA hot-swap** is not supported on either platform — each LoRA
  needs to be merged into a base checkpoint and shipped as its own
  artifact. Watercolor doesn't need a LoRA, so the no-LoRA path is the
  cheapest aesthetic to ship.

## Repository layout

```
local-ai-rnd/
  data/sample_week.json             synthetic 7-day record (no real user data)
  src/diary.py                      load + summarize the synthetic week
  src/prompt_builder.py             WeekSummary -> SD prompt
  src/base_collage.py               WeekSummary -> PIL collage (img2img seed)
  src/generator.py                  SD-Turbo txt2img + img2img wrapper
  src/styles.py                     aesthetic presets (watercolor, riso, ...)
  src/app.py                        Gradio UI
  scripts/refresh_samples.py        regenerate the SD-Turbo reference images
  scripts/quality_demo.py           regenerate the 30-step quality references
  scripts/style_explore.py          regenerate the aesthetic-preset references
  scripts/cross_platform_demo.py    regenerate the cross-platform references
  docs/mobile-architecture.md       on-device deployment plan
  samples/                          committed reference outputs
```

## Run it

Windows (PowerShell), Python 3.9+:

```powershell
# 1. (one-time) create venv + install deps
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. launch the Gradio UI locally
.\.venv\Scripts\python.exe -m src.app

# 2b. or expose a public *.gradio.live tunnel (link expires in ~72h, anyone
#     with the URL can use it — don't leave it running unattended)
.\.venv\Scripts\python.exe -m src.app --share
```

Then open http://127.0.0.1:7860.

To regenerate any of the reference image groups in `samples/`:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_samples.py        # SD-Turbo references
.\.venv\Scripts\python.exe scripts\quality_demo.py           # 30-step references
.\.venv\Scripts\python.exe scripts\style_explore.py          # aesthetic presets
.\.venv\Scripts\python.exe scripts\cross_platform_demo.py    # cross-platform
```

### First run

The first invocation downloads the relevant model weights from Hugging
Face into the local cache (~1.4 GB for SD-Turbo, ~4 GB per SD 1.5
fine-tune). Subsequent runs are cached.

### Optional GPU acceleration on AMD/Intel (Windows)

```powershell
.\.venv\Scripts\python.exe -m pip install torch-directml
$env:LAR_DEVICE = "directml"
.\.venv\Scripts\python.exe -m src.app
```

DirectML on the Radeon should be ~5-10× faster than the CPU numbers
quoted above.

### Environment knobs

| Variable        | Default                      | Notes                                 |
|-----------------|------------------------------|---------------------------------------|
| `LAR_MODEL_ID`  | `stabilityai/sd-turbo`       | Override for the Gradio app's default |
| `LAR_STEPS`     | `2`                          | SD-Turbo is happy at 1-4              |
| `LAR_GUIDANCE`  | `0.0`                        | SD-Turbo is trained with guidance off |
| `LAR_STRENGTH`  | `0.85`                       | img2img: lower = closer to base       |
| `LAR_DEVICE`    | auto (cuda > directml > cpu) | Force `cpu` or `directml`             |
