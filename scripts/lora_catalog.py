"""LoRA catalog — 5 diverse events × 17 LoRAs grid (= 85 cells).

Events span indoor (cafe, library, home living room) and outdoor
(market, riverside) settings, different times of day, different
weather, different seasons. Each LoRA is fused at scale 0.9, with
its Civitai-official trigger auto-prepended via LORA_TRIGGERS.

Resumes safely — each cell saves an output PNG immediately and the
final markdown index is rebuilt at the end. Re-running skips cells
whose PNG already exists, so this can be interrupted and continued.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr so Korean comments and em-dashes don't crash
# the script on Windows cp949 consoles when piped through `tee`.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt, LORA_TRIGGERS  # noqa: E402

SAMPLES = ROOT / "samples"
LORA_DIR = ROOT / "models" / "lora"

# 5 deliberately diverse events — 2 indoor + 3 outdoor, spanning
# morning to night, four seasons, four weather types.
EVENTS: list[tuple[str, dict]] = [
    ("e1_morning_home_livingroom", {
        # weekday morning, sunny spring, indoor living room
        "time": "08:00", "weather": "맑음", "date": "2026-04-12",
        "country": "대한민국", "city": "서울시", "place": "거실",
    }),
    ("e2_evening_cafe_rain", {
        # rainy autumn evening, indoor cafe
        "time": "20:00", "weather": "비", "date": "2026-10-08",
        "country": "대한민국", "city": "서울시", "place": "광화문 카페",
    }),
    ("e3_summer_river_night", {
        # summer night by the river, outdoor
        "time": "22:00", "weather": "맑음", "date": "2026-07-20",
        "country": "대한민국", "city": "서울시", "place": "한강공원",
    }),
    ("e4_winter_market_noon", {
        # cold snowy noon at a Korean traditional market, outdoor
        "time": "12:30", "weather": "눈", "date": "2026-01-15",
        "country": "대한민국", "city": "서울시", "place": "광장시장",
    }),
    ("e5_indoor_library_afternoon", {
        # afternoon at the library, overcast, indoor
        "time": "15:00", "weather": "흐림", "date": "2026-09-25",
        "country": "대한민국", "city": "서울시", "place": "도서관",
    }),
]

# (visual_style key in LORA_TRIGGERS, weight filename in models/lora/, scale)
LORAS: list[tuple[str, str, float]] = [
    ("worstimever",                "worstimever_xl.safetensors",         0.9),
    ("mspaint_portraits",          "sdxl_mspaint_portraits.safetensors", 0.9),
    ("doodle_style",               "doodle_style.safetensors",           0.9),
    ("doodles_in_real_life",       "doodles_in_real_life.safetensors",   0.9),
    ("cyberpunk_lines",            "cyberpunk_lines.safetensors",        0.9),
    ("linedrawing",                "linedrawing.safetensors",            0.9),
    ("lineart_zoolin",             "lineart_zoolin.safetensors",         0.9),
    ("asian_line_storyboard",      "asian_line_storyboard.safetensors",  0.9),
    ("soft_squishy_linework",      "soft_squishy_linework.safetensors",  0.9),
    ("colored_line",               "colored_line.safetensors",           0.9),
    ("clean_bw_line_art",          "clean_bw_line_art.safetensors",      0.9),
    ("tangbohu_landscape",         "tangbohu_landscape.safetensors",     0.9),
    ("digital_art_illustrations",  "digital_art_illustrations.safetensors", 0.9),
    ("simplex",                    "simplex.safetensors",                0.9),
    ("simple_toons",               "simple_toons.safetensors",           0.9),
    ("jackledead_artstyle",        "jackledead_artstyle.safetensors",    0.9),
    ("japanese_illustration",      "japanese_illustration.safetensors",  0.9),
]

DIFFUSION_SEED = 42  # fixed across all cells so the only variation is LoRA + event
PROMPT_SEED   = 7    # fixed phrase pick per event so cells are comparable across LoRAs


def existing_loras() -> list[tuple[str, str, float]]:
    """Filter LORAS to those whose .safetensors file actually exists."""
    out: list[tuple[str, str, float]] = []
    missing: list[str] = []
    for label, fname, scale in LORAS:
        if (LORA_DIR / fname).exists():
            out.append((label, fname, scale))
        else:
            missing.append(fname)
    if missing:
        print(f"[cat] missing {len(missing)} LoRA file(s) — skipping:")
        for m in missing:
            print(f"[cat]   - {m}")
    return out


def cell_path(lora_label: str, event_tag: str) -> Path:
    return SAMPLES / f"cat_{lora_label}__{event_tag}.png"


def main() -> None:
    print("[cat] loading SDXL-Turbo (cached)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    loras = existing_loras()
    total = len(loras) * len(EVENTS)
    print(f"[cat] {len(loras)} LoRAs × {len(EVENTS)} events = {total} cells")

    done = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []

    for li, (lora_label, lora_file, scale) in enumerate(loras):
        print(f"\n[cat] === ({li+1}/{len(loras)}) fusing {lora_label} ===")

        try: pipe.unfuse_lora()
        except Exception: pass
        try: pipe.unload_lora_weights()
        except Exception: pass

        try:
            pipe.load_lora_weights(
                str(LORA_DIR), weight_name=lora_file, adapter_name=lora_label
            )
            pipe.fuse_lora(lora_scale=scale)
        except Exception as e:
            print(f"[cat]   [FAIL] load_lora_weights: {type(e).__name__}: {e}")
            for ev_tag, _ev in EVENTS:
                failed.append((lora_label, ev_tag, f"lora load failed: {e}"))
            continue

        for ev_tag, ev in EVENTS:
            out_path = cell_path(lora_label, ev_tag)
            done += 1
            if out_path.exists() and out_path.stat().st_size > 1_000:
                skipped += 1
                print(f"[cat] ({done}/{total}) {lora_label} / {ev_tag} — already exists, skipping")
                continue

            try:
                built = build_event_prompt(
                    ev, visual_style=lora_label, prompt_seed=PROMPT_SEED
                )
            except Exception as e:
                print(f"[cat]   [FAIL] prompt build: {e}")
                failed.append((lora_label, ev_tag, f"prompt build: {e}"))
                continue

            print(f"[cat] ({done}/{total}) --- {lora_label} / {ev_tag} ---")
            print(f"[cat]   prompt: {built.positive[:140]}...")

            t = time.time()
            try:
                img = pipe(
                    prompt=built.positive,
                    negative_prompt=built.negative,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    width=512,
                    height=512,
                    generator=torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED),
                ).images[0]
                img.save(out_path)
                print(f"[cat]   {time.time() - t:.1f}s -> {out_path.name}")
            except Exception as e:
                print(f"[cat]   [FAIL] generate: {type(e).__name__}: {e}")
                failed.append((lora_label, ev_tag, f"generate: {e}"))

    # Build the markdown index regardless of whether all cells succeeded
    write_index(loras, EVENTS, failed)

    print(f"\n[cat] {done} attempted, {skipped} pre-existing")
    if failed:
        print(f"[cat] {len(failed)} failures:")
        for lora, ev, msg in failed:
            print(f"[cat]   - {lora} / {ev}: {msg}")


def write_index(
    loras: list[tuple[str, str, float]],
    events: list[tuple[str, dict]],
    failed: list[tuple[str, str, str]],
) -> None:
    md_lines: list[str] = [
        "# LoRA catalog — diverse events × multiple style LoRAs",
        "",
        "Same SDXL-Turbo, same diffusion seed (42), same `prompt_seed=7`",
        "(so all LoRAs see the same data prompt for a given event). Each",
        "LoRA fused at scale 0.9 with its Civitai-official trigger",
        "auto-prepended via `LORA_TRIGGERS`.",
        "",
        "Five events span indoor and outdoor, four seasons, four weather",
        "types, morning through night.",
        "",
    ]
    # Event header table
    md_lines.append("| event | data |")
    md_lines.append("| --- | --- |")
    for ev_tag, ev in events:
        md_lines.append(
            f"| `{ev_tag}` | {ev['time']} / {ev['weather']} / "
            f"{ev['date']} / {ev['place']} |"
        )
    md_lines.append("")

    # Per-LoRA section: 5-cell row + prompts list
    failed_set = {(l, e) for l, e, _ in failed}
    for li, (lora_label, lora_file, scale) in enumerate(loras):
        trig = LORA_TRIGGERS.get(lora_label, "")
        md_lines.append(f"## {lora_label}  (trigger: `{trig or '<none>'}`)")
        md_lines.append("")
        # Image grid
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
        # Per-cell prompts (re-derived deterministically using the same
        # PROMPT_SEED — these match exactly what was sent to SDXL).
        md_lines.append("**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):")
        md_lines.append("")
        for ev_tag, ev in events:
            try:
                ep = build_event_prompt(
                    ev, visual_style=lora_label, prompt_seed=PROMPT_SEED
                )
                # Replace pipe with HTML entity so it doesn't break the
                # parent table if anyone embeds this section into one.
                p_text = ep.positive.replace("|", "&#124;")
            except Exception as e:
                p_text = f"(prompt build failed: {e})"
            md_lines.append(f"- **`{ev_tag}`** — {p_text}")
        md_lines.append("")

    (SAMPLES / "lora_catalog.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\n[cat] index -> samples/lora_catalog.md")


if __name__ == "__main__":
    if "--index-only" in sys.argv:
        # Skip generation; just rebuild samples/lora_catalog.md from
        # whatever PNGs already exist on disk. Useful after the script's
        # write_index function is changed and you want to refresh the
        # markdown without regenerating images.
        write_index(existing_loras(), EVENTS, [])
    else:
        main()
