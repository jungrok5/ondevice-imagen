"""SD 1.5 LoRA compare — same 5 events as lora_catalog.py, but on a
SD 1.5 base so LoRAs that only ship for SD 1.5 (Drawing 110244, Inked
Portrait 947591) actually load.

Why a separate script: lora_catalog.py uses SDXL-Turbo and rejects
SD 1.5 LoRAs with a module-mismatch error at load_lora_weights. The
two new candidates the user asked us to evaluate are SD 1.5 only, so
we need a parallel evaluation track on a matching base.

Output:
- samples/sd15_compare_<lora>__<event>.png  (10 cells)
- samples/sd15_compare_grid.md              (markdown index)

Resumes safely (skips cells whose PNG already exists, like the SDXL
catalog script).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt, LORA_TRIGGERS  # noqa: E402

SAMPLES = ROOT / "samples"
LORA_DIR = ROOT / "models" / "lora"

# Re-use the exact same 5 events lora_catalog.py uses so the SDXL grid
# and this SD 1.5 grid are directly comparable cell-for-cell.
EVENTS: list[tuple[str, dict]] = [
    ("e1_morning_home_livingroom", {
        "time": "08:00", "weather": "맑음", "date": "2026-04-12",
        "country": "대한민국", "city": "서울시", "place": "거실",
    }),
    ("e2_evening_cafe_rain", {
        "time": "20:00", "weather": "비", "date": "2026-10-08",
        "country": "대한민국", "city": "서울시", "place": "광화문 카페",
    }),
    ("e3_summer_river_night", {
        "time": "22:00", "weather": "맑음", "date": "2026-07-20",
        "country": "대한민국", "city": "서울시", "place": "한강공원",
    }),
    ("e4_winter_market_noon", {
        "time": "12:30", "weather": "눈", "date": "2026-01-15",
        "country": "대한민국", "city": "서울시", "place": "광장시장",
    }),
    ("e5_indoor_library_afternoon", {
        "time": "15:00", "weather": "흐림", "date": "2026-09-25",
        "country": "대한민국", "city": "서울시", "place": "도서관",
    }),
]

# (visual_style key in LORA_TRIGGERS, .safetensors filename, lora scale)
# Per-LoRA scales follow each Civitai page's recommendation when given,
# else fall back to 0.8 (drawing 110244's example) or 0.9 (the SDXL
# catalog default).
LORAS: list[tuple[str, str, float]] = [
    ("sd15_drawing_nty",         "sd15_drawing_nty.safetensors",         0.8),
    ("sd15_inked_portrait",      "sd15_inked_portrait.safetensors",      0.9),
    ("sd15_paintstyle",          "sd15_paintstyle.safetensors",          0.8),
    ("sd15_oil_painting_stick",  "sd15_oil_painting_stick.safetensors",  0.8),
    ("sd15_dwmpainting",         "sd15_dwmpainting.safetensors",         0.8),
    ("sd15_oil_painting_feel",   "sd15_oil_painting_feel.safetensors",   0.8),  # author: <1.0
    ("sd15_ms_paint",            "sd15_ms_paint.safetensors",            0.9),
    ("sd15_oil_painting_style",  "sd15_oil_painting_style.safetensors",  0.8),
    ("sd15_caravaggio",          "sd15_caravaggio.safetensors",          0.6),  # author: 0.4-0.7
    ("sd15_classic_oil_painting","sd15_classic_oil_painting.safetensors", 0.8),
    ("sd15_light_oil_painting",  "sd15_light_oil_painting.safetensors",  0.8),
    ("sd15_disco_brush",         "sd15_disco_brush.safetensors",         1.0),  # author: 1.0-2.0, conservative
    ("sd15_disco_rostov",        "sd15_disco_rostov.safetensors",        0.9),  # author: 0.6-0.9 emphasize
]

DIFFUSION_SEED = 42
PROMPT_SEED   = 7
STEPS         = 12      # DPM++ 2M can hit decent quality at 12 on SD 1.5
CFG           = 7.0     # matches the Civitai example image for 110244
WIDTH         = 512
HEIGHT        = 512
BASE_MODEL    = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def cell_path(lora_label: str, event_tag: str) -> Path:
    return SAMPLES / f"sd15_compare_{lora_label}__{event_tag}.png"


def main() -> None:
    print(f"[sd15] loading {BASE_MODEL} (cached)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # DPMSolverMultistep ≈ "DPM++ 2M Karras" in the Civitai example image.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )
    pipe = pipe.to("cpu")

    total = len(LORAS) * len(EVENTS)
    print(f"[sd15] {len(LORAS)} LoRAs × {len(EVENTS)} events = {total} cells")
    print(f"[sd15] {STEPS} steps, CFG {CFG}, {WIDTH}x{HEIGHT}, seed {DIFFUSION_SEED}")

    done = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []

    for li, (lora_label, lora_file, scale) in enumerate(LORAS):
        if not (LORA_DIR / lora_file).exists():
            print(f"[sd15] missing {lora_file} — skipping all events for {lora_label}")
            for ev_tag, _ev in EVENTS:
                failed.append((lora_label, ev_tag, "lora file missing"))
            continue

        print(f"\n[sd15] === ({li+1}/{len(LORAS)}) fusing {lora_label} (scale={scale}) ===")
        try: pipe.unfuse_lora()
        except Exception: pass
        try: pipe.unload_lora_weights()
        except Exception: pass

        try:
            pipe.load_lora_weights(
                str(LORA_DIR), weight_name=lora_file, adapter_name=lora_label,
            )
            pipe.fuse_lora(lora_scale=scale)
        except Exception as e:
            print(f"[sd15]   [FAIL] load_lora_weights: {type(e).__name__}: {e}")
            for ev_tag, _ev in EVENTS:
                failed.append((lora_label, ev_tag, f"lora load failed: {e}"))
            continue

        for ev_tag, ev in EVENTS:
            out_path = cell_path(lora_label, ev_tag)
            done += 1
            if out_path.exists() and out_path.stat().st_size > 1_000:
                skipped += 1
                print(f"[sd15] ({done}/{total}) {lora_label} / {ev_tag} — already exists, skipping")
                continue

            try:
                built = build_event_prompt(
                    ev, visual_style=lora_label, prompt_seed=PROMPT_SEED,
                )
            except Exception as e:
                print(f"[sd15]   [FAIL] prompt build: {e}")
                failed.append((lora_label, ev_tag, f"prompt build: {e}"))
                continue

            print(f"[sd15] ({done}/{total}) --- {lora_label} / {ev_tag} ---")
            print(f"[sd15]   prompt: {built.positive[:140]}...")

            t = time.time()
            try:
                img = pipe(
                    prompt=built.positive,
                    negative_prompt=built.negative,
                    num_inference_steps=STEPS,
                    guidance_scale=CFG,
                    width=WIDTH,
                    height=HEIGHT,
                    generator=torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED),
                ).images[0]
                img.save(out_path)
                print(f"[sd15]   {time.time() - t:.1f}s -> {out_path.name}")
            except Exception as e:
                print(f"[sd15]   [FAIL] generate: {type(e).__name__}: {e}")
                failed.append((lora_label, ev_tag, f"generate: {e}"))

    write_index(LORAS, EVENTS, failed)

    print(f"\n[sd15] {done} attempted, {skipped} pre-existing")
    if failed:
        print(f"[sd15] {len(failed)} failures:")
        for lora, ev, msg in failed:
            print(f"[sd15]   - {lora} / {ev}: {msg}")


def write_index(
    loras: list[tuple[str, str, float]],
    events: list[tuple[str, dict]],
    failed: list[tuple[str, str, str]],
) -> None:
    md_lines: list[str] = [
        f"# SD 1.5 LoRA compare — {len(loras)} candidates × {len(events)} events",
        "",
        f"Same 5 events as `lora_catalog.md`. Base: `{BASE_MODEL}`",
        f"(SD 1.5 vanilla, not Turbo — these LoRAs are SD 1.5 only).",
        "",
        f"Sampler: DPM++ 2M Karras, **{STEPS} steps**, CFG **{CFG}**, ",
        f"{WIDTH}x{HEIGHT}, fixed seed **{DIFFUSION_SEED}**, "
        f"`prompt_seed={PROMPT_SEED}` so every cell sees the same data prompt for a given event.",
        "",
        "Per-LoRA fuse scale follows each Civitai page's recommendation when",
        "the author specified one (e.g. `caravaggio` 0.6 for the 0.4-0.7",
        "range, `disco_brush` 1.0 conservative end of 1.0-2.0); otherwise",
        "0.8 (drawing 110244's example image weight) or 0.9 (catalog default).",
        "",
    ]

    md_lines.append("| event | data |")
    md_lines.append("| --- | --- |")
    for ev_tag, ev in events:
        md_lines.append(
            f"| `{ev_tag}` | {ev['time']} / {ev['weather']} / "
            f"{ev['date']} / {ev['place']} |"
        )
    md_lines.append("")

    failed_set = {(l, e) for l, e, _ in failed}
    for li, (lora_label, lora_file, scale) in enumerate(loras):
        trig = LORA_TRIGGERS.get(lora_label, "")
        md_lines.append(f"## {lora_label}  (trigger: `{trig or '<none>'}`, scale {scale})")
        md_lines.append("")
        md_lines.append("| " + " | ".join(t for t, _ in events) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(events)) + " |")
        cells = []
        for ev_tag, _ev in events:
            if (lora_label, ev_tag) in failed_set:
                cells.append("(failed)")
            else:
                p = cell_path(lora_label, ev_tag)
                if p.exists():
                    cells.append(f"![]({p.name})")
                else:
                    cells.append("(missing)")
        md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")
        md_lines.append("**Prompts** (CLIP truncates at 77 tokens):")
        md_lines.append("")
        for ev_tag, ev in events:
            try:
                ep = build_event_prompt(
                    ev, visual_style=lora_label, prompt_seed=PROMPT_SEED,
                )
                p_text = ep.positive.replace("|", "&#124;")
            except Exception as e:
                p_text = f"(prompt build failed: {e})"
            md_lines.append(f"- **`{ev_tag}`** — {p_text}")
        md_lines.append("")

    (SAMPLES / "sd15_compare_grid.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8",
    )
    print(f"\n[sd15] index -> samples/sd15_compare_grid.md")


if __name__ == "__main__":
    if "--index-only" in sys.argv:
        write_index(LORAS, EVENTS, [])
    else:
        main()
