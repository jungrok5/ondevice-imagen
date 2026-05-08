"""5 user-input variations × 2 style LoRAs = 10-cell grid.

Uses the refactored src/event_prompt.py with auto-trigger prepending
via the LORA_TRIGGERS registry. Per the testing rule, every cell that
loads a LoRA includes that LoRA's card-specified trigger in the prompt.

Same SDXL-Turbo base, same seed across all 10 cells (so the only
varying axes are the event data and the visual_style).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt, LORA_TRIGGERS  # noqa: E402

SAMPLES = ROOT / "samples"
LORA_DIR = ROOT / "models" / "lora"

EVENTS = [
    ("e1_xmas_evening", {
        "time": "19:00", "weather": "맑음", "date": "2026-12-25",
        "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
    }),
    ("e2_summer_park", {
        "time": "14:30", "weather": "맑음", "date": "2026-08-15",
        "country": "대한민국", "city": "서울시", "place": "한강 공원",
    }),
    ("e3_rainy_cafe", {
        "time": "08:45", "weather": "비", "date": "2026-03-20",
        "country": "대한민국", "city": "서울시", "place": "광화문 카페",
    }),
    ("e4_autumn_night_home", {
        "time": "23:30", "weather": "흐림", "date": "2026-10-31",
        "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
    }),
    ("e5_snow_noon_market", {
        "time": "12:00", "weather": "눈", "date": "2026-02-04",
        "country": "대한민국", "city": "서울시", "place": "광장시장",
    }),
]

LORA_FILES = {
    "worstimever":       "worstimever_xl.safetensors",
    "mspaint_portraits": "sdxl_mspaint_portraits.safetensors",
}

SEED = 42


def main() -> None:
    print("[var] loading SDXL-Turbo (cached)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    md_lines = [
        "# 5 event variations × 2 style LoRAs",
        "",
        "Same SDXL-Turbo, same seed (42). Each row is one event; columns",
        "are the two style LoRAs (with their triggers auto-prepended via",
        "src/event_prompt.py LORA_TRIGGERS registry).",
        "",
        "| event | data summary | worstimever | mspaint_portraits |",
        "| --- | --- | --- | --- |",
    ]

    for ev_tag, ev in EVENTS:
        # Pre-compute the breakdown once for the row label
        ep_w = build_event_prompt(ev, visual_style="worstimever")
        moments = ", ".join(p for _src, p in ep_w.breakdown)
        data_summary = (
            f"{ev['time']} / {ev['weather']} / {ev['date']} / {ev['place']}"
        )
        cell_paths: dict[str, str] = {}

        for style_name, lora_file in LORA_FILES.items():
            print(f"\n[var] === {ev_tag} × {style_name} ===")

            try: pipe.unfuse_lora()
            except Exception: pass
            try: pipe.unload_lora_weights()
            except Exception: pass

            pipe.load_lora_weights(
                str(LORA_DIR), weight_name=lora_file, adapter_name=style_name
            )
            pipe.fuse_lora(lora_scale=0.9)

            built = build_event_prompt(ev, visual_style=style_name)
            print(f"[var] prompt: {built.positive[:140]}...")

            t = time.time()
            img = pipe(
                prompt=built.positive,
                negative_prompt=built.negative,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=512,
                height=512,
                generator=torch.Generator(device="cpu").manual_seed(SEED),
            ).images[0]
            out_path = SAMPLES / f"var_{ev_tag}_{style_name}.png"
            img.save(out_path)
            print(f"[var]   {time.time() - t:.1f}s -> {out_path.name}")
            cell_paths[style_name] = out_path.name

        md_lines.append(
            f"| `{ev_tag}` | {data_summary} "
            f"| ![]({cell_paths.get('worstimever', '')}) "
            f"| ![]({cell_paths.get('mspaint_portraits', '')}) |"
        )

    (SAMPLES / "var_grid.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\n[var] index -> samples/var_grid.md")


if __name__ == "__main__":
    main()
