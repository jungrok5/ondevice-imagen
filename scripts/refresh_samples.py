"""Regenerate the committed reference images in samples/.

Runs: base_collage -> img2img -> txt2img and writes PNGs into samples/.
First invocation downloads the SD-Turbo weights (~1.4 GB).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder, base_collage, generator  # noqa: E402

SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

week = diary.load_week(ROOT / "data" / "sample_week.json")
summary = diary.summarize(week)
built = prompt_builder.build(summary)

print("[refresh] summary :", diary.summary_to_text(summary))
print("[refresh] prompt  :", built.positive[:120], "...")

print("[refresh] rendering base collage")
collage = base_collage.render(summary)
collage.save(SAMPLES / "base_collage.png")

print("[refresh] running img2img (this will download SD-Turbo on first run)")
t0 = time.time()
img2img = generator.generate_img2img(
    prompt=built.positive,
    negative_prompt=built.negative,
    init_image=collage,
    steps=4,
    strength=0.99,
    seed=42,
)
img2img.save(SAMPLES / "sd_img2img.png")
print(f"[refresh] img2img done in {time.time() - t0:.1f}s")

print("[refresh] running txt2img for comparison")
t0 = time.time()
txt2img = generator.generate_txt2img(
    prompt=built.positive,
    negative_prompt=built.negative,
    steps=4,
    seed=42,
)
txt2img.save(SAMPLES / "sd_txt2img.png")
print(f"[refresh] txt2img done in {time.time() - t0:.1f}s")

print(f"[refresh] all samples written to: {SAMPLES}")
