"""v10 — the actual product flow.

Real spec (per user, after iteration):
  input:  diary text — time / weather / place(POI)
  output: SD generates an UNEXPECTED 하찮은-style scene from that text
  no photo input
  no fixed seed — each call produces a different result so user gets
                  a surprise even if their week looks similar to last
                  week's

Pipeline:
  diary data  -> prompt_builder.build()  (no seed → random phrases)
              -> append worstimever LoRA trigger
              -> SDXL-Turbo txt2img with worstimever LoRA fused
              -> finished (no jitter — user feedback)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder  # noqa: E402
from scripts.random_weeks_doodle import make_random_week  # noqa: E402

SAMPLES = ROOT / "samples"
LORA_PATH = ROOT / "models" / "lora" / "worstimever_xl.safetensors"

# DD-wte trigger phrase per the LoRA card. Combined with our existing
# protagonist-rotation phrases, this should push the styled output into
# the "deliberately worst" register the trend wants.
LORA_TRIGGER = "DD-wte artstyle, worst-im-ever cartoon doodle"

NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)


def main() -> None:
    print(f"[v10] loading SDXL-Turbo + worstimever LoRA...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    pipe.load_lora_weights(
        str(LORA_PATH.parent), weight_name=LORA_PATH.name
    )
    pipe.fuse_lora(lora_scale=0.9)
    print(f"[v10] LoRA fused. Generating...")

    md = [
        "# v10 — SDXL-Turbo + worstimever LoRA, txt2img only",
        "",
        "Product flow: diary text -> prompt_builder -> SDXL-Turbo + LoRA",
        "-> result. No photo input. Seed is random — same data twice gives",
        "different unexpected outputs.",
        "",
        "| seed | week shape | prompt | output |",
        "| --- | --- | --- | --- |",
    ]

    # Three different random week shapes (matching the four protagonist
    # axes from earlier rounds): cup-heavy, time-morning, place-cafe,
    # weather-rainy, plus one variety pair on the same data.
    runs = [
        ("w11_v1", 11),  # weather-rainy
        ("w22_v1", 22),  # time-morning
        ("w33_v1", 33),  # place-cafe
        ("w44_v1", 44),  # cup-heavy
        ("w22_v2", 22),  # same data as w22_v1 → must produce different image
    ]

    for tag, week_seed in runs:
        week = make_random_week(week_seed)
        summary = diary.summarize(week)
        # seed=None → fresh phrase selection per call
        built = prompt_builder.build(summary)
        prompt = f"{LORA_TRIGGER}, {built.positive}"
        print(f"\n[v10] {tag} (week seed {week_seed}):")
        print(f"      protagonist={built.protagonist}")
        print(f"      moments={built.subjects}")

        t = time.time()
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            # generator left None → torch picks fresh randomness each call
        )
        img = out.images[0]
        path = SAMPLES / f"v10_{tag}.png"
        img.save(path)
        print(f"[v10]   {time.time() - t:.1f}s -> {path.name}")

        # Truncate prompt for table
        short = built.positive[:80] + ("..." if len(built.positive) > 80 else "")
        md.append(
            f"| {tag} | seed={week_seed} {built.protagonist} "
            f"| `{short}` | ![]({path.name}) |"
        )

    (SAMPLES / "match_v10_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[v10] index -> samples/match_v10_grid.md")


if __name__ == "__main__":
    main()
