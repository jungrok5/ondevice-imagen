"""Demonstrate per-call variability on the same data.

Holds the diary fixed (seed 22 = place_cafe week) and runs the
generation pipeline four times *without* fixing any RNG. Each call
gets fresh random state in:
  - prompt_builder.build()        : phrase pool selection
  - generator.generate_txt2img()  : SD initial noise

So the protagonist axis stays the same (the data is the same), but the
phrasing and the image both vary call-to-call. Every output also gets a
kasun pass (light) applied so the 하찮은 finish lands.

This is what the actual app should feel like: same Sunday morning, same
weekly summary, different postcard — surprise even when life is routine.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder, generator  # noqa: E402
from src.kasun_filter import kasun  # noqa: E402
from src.pixelize import pixelize  # noqa: E402
from scripts.random_weeks_doodle import make_random_week  # noqa: E402


SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


WEEK_SEED = 22  # place_cafe shape
N_VARIANTS = 4


def main() -> None:
    week = make_random_week(WEEK_SEED)
    summary = diary.summarize(week)
    print(f"[variety] week_seed={WEEK_SEED}")
    print(f"[variety] summary: {diary.summary_to_text(summary)}")
    print(f"[variety] places: {summary.place_categories}")

    for i in range(1, N_VARIANTS + 1):
        # No seed → fresh RNG every call
        built = prompt_builder.build(summary)
        print(f"\n[variety] variant {i}")
        print(f"  protagonist: {built.protagonist}")
        print(f"  moments    : {built.subjects}")

        t = time.time()
        img = generator.generate_txt2img(
            prompt=built.positive,
            negative_prompt=built.negative,
            steps=4,
            guidance=0.0,
            width=512,
            height=512,
            seed=None,  # truly random
        )
        raw_path = SAMPLES / f"variety_w{WEEK_SEED}_v{i}.png"
        img.save(raw_path)
        print(f"  raw   -> {raw_path.name}  ({time.time() - t:.1f}s)")

        # Apply kasun light (the closest match to GPT-4o 하찮은 result)
        kasun_img = kasun(img, grid=64, colors=2, output_size=1024)
        kasun_path = SAMPLES / f"variety_w{WEEK_SEED}_v{i}_kasun.png"
        kasun_img.save(kasun_path)
        print(f"  kasun -> {kasun_path.name}")

        # Also pixel-art (16-color) for color variant lovers
        pixel_img = pixelize(img, grid=128, colors=16, output_size=1024)
        pixel_path = SAMPLES / f"variety_w{WEEK_SEED}_v{i}_pixel.png"
        pixel_img.save(pixel_path)
        print(f"  pixel -> {pixel_path.name}")

    print("\n[variety] done")


if __name__ == "__main__":
    main()
