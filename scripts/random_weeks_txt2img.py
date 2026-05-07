"""Same random weeks, but pure txt2img — no collage seed.

This isolates the prompt-driven variation from the layout-driven variation.
Each week gets its own noise seed so the SD-Turbo random init differs too.

Output: samples/random_w<seed>_txt2img.png        raw SD output
        samples/random_w<seed>_txt2img_pixel.png  pixelize at 1024x1024

Same SD-Turbo settings the doodle path uses (4 steps, guidance 0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder, generator  # noqa: E402
from src.pixelize import pixelize  # noqa: E402

# Reuse the same synthetic-week generator
from scripts.random_weeks_doodle import make_random_week  # noqa: E402


SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


def run_one(seed: int) -> dict:
    week = make_random_week(seed)
    summary = diary.summarize(week)
    built = prompt_builder.build(summary)

    print(f"\n[t2i] seed={seed}  cups_in_prompt={summary.water_event_count}")
    print(f"[t2i]   subjects={built.subjects}")

    t = time.time()
    img = generator.generate_txt2img(
        prompt=built.positive,
        negative_prompt=built.negative,
        steps=4,
        guidance=0.0,
        width=512,
        height=512,
        seed=seed,  # use the week seed itself as noise seed
    )
    raw_path = SAMPLES / f"random_w{seed}_txt2img.png"
    img.save(raw_path)
    print(f"[t2i]   raw -> {raw_path.name}  ({time.time() - t:.1f}s)")

    pixel = pixelize(img, grid=64, colors=16, output_size=1024)
    pixel_path = SAMPLES / f"random_w{seed}_txt2img_pixel.png"
    pixel.save(pixel_path)
    print(f"[t2i]   pixel -> {pixel_path.name}")

    return {
        "seed": seed,
        "summary": diary.summary_to_text(summary),
        "subjects": built.subjects,
        "prompt": built.positive,
        "raw": raw_path.name,
        "pixel": pixel_path.name,
    }


def main() -> None:
    rows = []
    for s in [11, 22, 33, 44]:
        rows.append(run_one(s))
    print("\n[t2i] all done")


if __name__ == "__main__":
    main()
