# local-ai-rnd — Weekly Postcard Doodle

PC prototype for an Android/iOS app that turns a week of personal data
(water intake, weather, GPS-derived POIs) into a deliberately badly drawn
postcard, generated on-device.

The aesthetic target (per the user's spec, in Korean):

> 첨부한 이미지를 최대한 서툴고, 휘갈긴 듯하고, 진짜 한심하게 다시 그려줘.
> 배경은 흰색으로 하고, 옛날 컴퓨터 그림판 프로그램에서 마우스로 그린 것처럼
> 보이게 해줘. ... 픽셀 하나하나 보이는 저화질 느낌도 살려서 ...

That intent is encoded as SD-friendly tags in [src/prompt_builder.py](src/prompt_builder.py).

## What this prototype answers

1. Can a small Stable Diffusion model produce the "intentionally bad doodle"
   aesthetic from a structured weekly summary?
2. Does img2img on a rule-rendered collage beat pure txt2img?
3. What latency / model size do we need to budget for the mobile port?

The full mobile architecture lives in [docs/mobile-architecture.md](docs/mobile-architecture.md).

## Repository layout

```
local-ai-rnd/
  data/sample_week.json         7 days of fake diary data
  src/diary.py                  load + summarize a week
  src/prompt_builder.py         WeekSummary -> SD prompt
  src/base_collage.py           WeekSummary -> PIL collage (img2img seed)
  src/generator.py              SD-Turbo txt2img + img2img
  src/app.py                    Gradio UI for end-to-end testing
  outputs/                      generated PNGs
  docs/mobile-architecture.md   the on-device target
```

## Run it

Windows (PowerShell), Python 3.9+:

```powershell
# 1. (already done by the bootstrap) create venv + install deps
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. launch the UI
.\.venv\Scripts\python.exe -m src.app
```

Then open http://127.0.0.1:7860 — the workflow is Summarize → img2img.

### First run

- The default model `stabilityai/sd-turbo` (~1.4 GB) downloads to the
  HuggingFace cache on first generation. Subsequent runs are instant.
- On CPU only, expect ~30-60 s per 512x512 image at 2 steps. On a GPU it
  drops to a few seconds.

### Optional GPU acceleration on AMD/Intel (Windows)

```powershell
.\.venv\Scripts\python.exe -m pip install torch-directml
$env:LAR_DEVICE = "directml"
.\.venv\Scripts\python.exe -m src.app
```

DirectML on a Radeon Pro 580X (4 GB) should be ~5-10x faster than CPU for
SD-Turbo at 512x512.

### Environment knobs

| Variable        | Default                       | Notes                                 |
|-----------------|-------------------------------|---------------------------------------|
| `LAR_MODEL_ID`  | `stabilityai/sd-turbo`        | Try `stabilityai/sdxl-turbo` if VRAM allows |
| `LAR_STEPS`     | `2`                           | SD-Turbo is happy at 1-4              |
| `LAR_GUIDANCE`  | `0.0`                         | SD-Turbo is trained with guidance off |
| `LAR_STRENGTH`  | `0.85`                        | img2img: lower = closer to base       |
| `LAR_DEVICE`    | auto (cuda > directml > cpu)  | Force `cpu` or `directml`             |

## How the prompt is built

`prompt_builder.build()` lifts subjects from the summary and concatenates
them with a fixed style block:

- water_event_count → "N mismatched water cups in a row"
- dominant_weather → "smiling sun" / "lumpy clouds" / etc.
- place categories → "wobbly coffee cup", "lopsided trees", ...
- style → "crude childlike doodle, MS Paint, lo-fi, pixelated, ..."

Tweak the dictionaries in `prompt_builder.py` to taste. Style tags are the
single biggest knob for "how bad" the result looks.

## What is *not* in the prototype

- Real data ingestion (HealthKit / Google Fit, weather API, geocoder). The
  prototype reads a JSON file; the mobile app will hydrate the same shape
  from on-device sources.
- Any LLM step. The prompt is rule-based on purpose — predictable and
  shippable. If you later want freer prose, hook the summary into an
  on-device LLM (Apple Intelligence FoundationModel / MediaPipe LLM).
- Multi-image postcard layout / typography. Out of scope for feasibility.
