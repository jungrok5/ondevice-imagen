"""Generate one postcard per aesthetic preset, all from the same diary.

Loads Dreamshaper-8 once (already cached locally) and reuses it for every
style — far cheaper than reloading per run. Same prompt subjects + same
seed across styles, so the only varying axis is the aesthetic.

Output: samples/style_<name>.png
Runtime on CPU: ~4 min/image (so ~12 min for 3 styles).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, styles  # noqa: E402

SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


# Subject block — written once, used across all styles. Lifts concrete scenes
# from data/sample_week.json instead of abstract tokens, because painterly
# styles want imagery, not adjectives.
def build_subject(week_path: Path) -> str:
    week = diary.load_week(week_path)
    s = diary.summarize(week)
    weather_phrase = {
        "sunny": "soft afternoon sunlight",
        "cloudy": "soft overcast light",
        "rainy": "rain on the window",
        "windy": "leaves drifting",
        "snowy": "gentle snow",
    }.get(s.dominant_weather, "soft daylight")

    place_phrase_map = {
        "cafe": "a cafe window",
        "park": "tree-lined park path",
        "work": "a desk by the window",
        "bookstore": "stacks of well-loved books",
        "restaurant": "a small noodle shop",
        "landmark": "a tower in the distance",
        "market": "a neighborhood market",
        "home": "a quiet kitchen table",
    }
    places = [place_phrase_map[c] for c in s.place_categories.keys() if c in place_phrase_map][:4]
    place_phrase = ", ".join(places) if places else "a quiet room"

    return (
        f"a hand-drawn postcard depicting one quiet week, "
        f"{weather_phrase}, {place_phrase}, "
        f"a row of teacups on a wooden table"
    )


# Pick three contrasting aesthetics so the comparison is meaningful.
STYLES_TO_RUN = [
    styles.WATERCOLOR_DIARY,
    styles.RISOGRAPH_ZINE,
    styles.CRAYON_PICTUREBOOK,
]


MODEL = "Lykon/dreamshaper-8"  # already cached from quality_demo.py
SEED = 7
STEPS = 30


def main() -> None:
    subject = build_subject(ROOT / "data" / "sample_week.json")
    print(f"[style] subject: {subject}\n")

    print(f"[style] loading {MODEL}")
    t_load = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    print(f"[style] loaded in {time.time() - t_load:.1f}s")

    for preset in STYLES_TO_RUN:
        out_path = SAMPLES / f"style_{preset.name}.png"
        if out_path.exists():
            print(f"[style] {preset.name}: skipped (exists)")
            continue

        positive, negative = styles.build_prompt(preset, subject)
        print(f"\n[style] === {preset.name} ===")
        print(f"[style] prompt: {positive[:120]}...")

        t = time.time()
        out = pipe(
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        out.images[0].save(out_path)
        print(f"[style] {preset.name} done in {time.time() - t:.1f}s -> {out_path}")

    print("\n[style] all done")


if __name__ == "__main__":
    main()
