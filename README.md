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

## Viral-aesthetic search

The "하찮은 프롬프트" trend went global in May 2025 because it stacked
five things at once: a meta-instruction (be deliberately inept),
recognizable-but-wrong output, earnest amateur energy, one-prompt
reusability, and a strong cultural rhyme (MS Paint).

[scripts/style_viral_search.py](scripts/style_viral_search.py) tries
five candidate aesthetics that share that DNA but explore different
cultural rhymes — to see whether SD 1.5 can deliver the same kind of
"recognizable + earnest + slightly wrong" charge through a single
short prompt. All five run on `dreamlike-art/dreamlike-diffusion-1.0`
(the most painterly fine-tune in the cross-platform table), same
subject + seed, 30 steps Euler-a (~4 min/image on CPU).

Same subject across all five: *"a young woman holding a coffee cup at
a cafe window, plants beside her"*.

| preset | output | reading |
|---|---|---|
| `renaissance_oil` | ![ren](samples/viral_renaissance_oil.png) | technically beautiful but the "elevate the mundane in oil paint" trope is already saturated. ★★★★ |
| `hotel_art_70s` | ![hotel](samples/viral_hotel_art_70s.png) | reads as *actual* 1970s mass-produced motel kitsch — wrong proportions, ugly palette, sad-but-charming. **★★★★★** |
| `embroidered_sampler` | ![sampler](samples/viral_embroidered_sampler.png) | cross-stitch face + scene, distinctive medium-as-message. Pixelation reads as "intentionally limited," warm grandma-craft DNA. **★★★★★** |
| `friends_mom_portrait` | ![mom](samples/viral_friends_mom_portrait.png) | closest cousin to 하찮은 프롬프트: same amateur-sincerity DNA in a *different cultural rhyme* (a 2003 colored-pencil sketch by a sincere suburban mom). **★★★★★** |
| `half_remembered_film` | ![film](samples/viral_half_remembered_film.png) | beautiful but aspirational — looks like a magazine ad, not a meme. Missing "wrongness." ★★★ |

**Reading the result space.** Three of the five actually carry the DNA:
`hotel_art_70s` and `embroidered_sampler` for their distinctive media,
`friends_mom_portrait` for being the most novel formulation — same
earnest-amateur energy as the original trend but in a different
medium and cultural reference. `renaissance_oil` lands but is already
heavily memed; `half_remembered_film` is too clean — no wrongness, no
hook.

### Round 2: drill into the two strongest finds

[scripts/style_viral_drill.py](scripts/style_viral_drill.py) runs four
variants — two on each of `friends_mom_portrait` and `hotel_art_70s` —
with the same subject and seed for direct comparison.

| variant | output | reading |
|---|---|---|
| `mom_korean_living_room` | ![drill1](samples/drill_mom_korean_living_room.png) | red wooden frame, dark navy starry background, green cardigan, plant — *reads as the kind of art on every Korean grandmother's living-room wall*. The frame is included **inside the image**, adding one more layer of meta to the "earnest hung-up amateur art" idea. **★★★★★ — round-2 winner, beats round-1.** |
| `retired_engineer_watercolor` | ![drill2](samples/drill_retired_engineer_watercolor.png) | competent loose watercolor, but "retired engineer" identity does not read visually — looks like generic amateur watercolor. ★★★★ |
| `korean_wedding_hall_90s` | ![drill3](samples/drill_korean_wedding_hall_90s.png) | the model is *too familiar* with this motif and renders it competently — loses the kitsch we wanted. ★★★ |
| `doctors_office_85` | ![drill4](samples/drill_doctors_office_85.png) | captures the soothing institutional palette but the hook is mild. ★★★★ |

**One real lesson.** Cultural specificity helps when the model has
*just enough* prior knowledge to recognize the rhyme but not enough to
render it perfectly. `mom_korean_living_room` hits the sweet spot —
recognizable but slightly off, with the framed-on-wall presentation
landing as a separate visual punchline. `korean_wedding_hall_90s`
overshoots into "competent" because the model has seen too many of
those.

### Current ranking

1. `mom_korean_living_room` — earnest amateur + Korean frame meta + globally legible
2. `friends_mom_portrait` — same DNA in a different cultural rhyme
3. `hotel_art_70s` — distinctive ugly-kitsch
4. `embroidered_sampler` — distinctive medium

The single most interesting candidate is now `mom_korean_living_room`:
the rhyme is specific enough to be funny but generic enough to apply
to any subject ("draw it like the painting on a Korean grandmother's
living-room wall, in its frame"), which is the four properties that
made the original 하찮은 프롬프트 portable across languages.

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
