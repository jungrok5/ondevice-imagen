"""Generate several random weekly diaries, run each through the doodle
pipeline, and save before/after pixel-perfect post-process for comparison.

For each random seed:
  1. Build a synthetic week (water log + weather + places).
  2. Build the prompt from it.
  3. Render the rule-based pictographic collage (img2img seed).
  4. Run SD-Turbo img2img (4 steps, strength 0.99) on the collage.
  5. Apply pixelize() to force a true pixel-perfect look.

Outputs per seed:
  samples/random_w<seed>_collage.png   the input collage
  samples/random_w<seed>_doodle.png    raw SD output
  samples/random_w<seed>_pixel.png     after pixelize (16 colors, 8x8 blocks)

Plus a single samples/random_weeks.md index page that lays everything
out side-by-side for reading on GitHub.

Runtime on CPU: ~30 s per seed (SD-Turbo is fast at 4 steps).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder, base_collage, generator  # noqa: E402
from src.pixelize import pixelize  # noqa: E402


SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


PLACES = [
    {"name": "Cafe", "category": "cafe"},
    {"name": "Park", "category": "park"},
    {"name": "Workplace", "category": "work"},
    {"name": "Bookstore", "category": "bookstore"},
    {"name": "Restaurant", "category": "restaurant"},
    {"name": "Tower", "category": "landmark"},
    {"name": "Market", "category": "market"},
    {"name": "Apartment", "category": "home"},
]

WEATHERS = ["sunny", "cloudy", "rainy", "windy", "snowy"]


def make_random_week(seed: int) -> dict:
    rng = random.Random(seed)

    # Pick a *dominant* weather for the week so the output prompt has a
    # clear top-1 instead of always being a wash.
    week_weather = rng.choice(WEATHERS)

    entries = []
    for d in range(7):
        # 0-4 sips per day; sometimes none, so totals vary across seeds
        n_water = rng.choices([0, 1, 2, 3, 4, 5], weights=[1, 2, 3, 3, 2, 1])[0]
        water = []
        for _ in range(n_water):
            h = rng.randint(7, 22)
            m = rng.randint(0, 59)
            ml = rng.choice([150, 200, 250, 300, 350])
            water.append({"time": f"{h:02d}:{m:02d}", "ml": ml})

        # 70% of days follow the dominant weather, 30% drift
        weather_today = (
            week_weather if rng.random() < 0.7 else rng.choice(WEATHERS)
        )
        weather = {"summary": weather_today, "temp_c": rng.randint(-5, 30), "icon": "x"}

        n_places = rng.randint(0, 3)
        places = rng.sample(PLACES, k=min(n_places, len(PLACES)))

        entries.append(
            {
                "date": f"2026-04-{27 + d}",
                "weather": weather,
                "water_intake": water,
                "places": places,
            }
        )

    return {
        "user_id": f"random_{seed}",
        "week_start": "2026-04-27",
        "week_end": "2026-05-03",
        "entries": entries,
    }


def run_one(seed: int) -> dict:
    week = make_random_week(seed)
    summary = diary.summarize(week)
    built = prompt_builder.build(summary)

    collage = base_collage.render(summary)
    collage_path = SAMPLES / f"random_w{seed}_collage.png"
    collage.save(collage_path)

    doodle = generator.generate_img2img(
        prompt=built.positive,
        negative_prompt=built.negative,
        init_image=collage,
        steps=4,
        strength=0.99,
        seed=42,
    )
    doodle_path = SAMPLES / f"random_w{seed}_doodle.png"
    doodle.save(doodle_path)

    # Keep grid=64 (chunky) for the SD-Turbo doodle vibe but render the PNG
    # at 1024 so it is properly visible.
    pixel = pixelize(doodle, grid=64, colors=16, output_size=1024)
    pixel_path = SAMPLES / f"random_w{seed}_pixel.png"
    pixel.save(pixel_path)

    return {
        "seed": seed,
        "summary": diary.summary_to_text(summary),
        "subjects": built.subjects,
        "prompt": built.positive,
        "collage": collage_path.name,
        "doodle": doodle_path.name,
        "pixel": pixel_path.name,
    }


def write_index(rows: list[dict]) -> None:
    """Write samples/random_weeks.md: a side-by-side reading layout."""
    lines = [
        "# Random weekly doodles",
        "",
        "Four randomized weekly diaries fed through the same pipeline:",
        "",
        "1. `make_random_week(seed)` synthesises a week of water/weather/places.",
        "2. `prompt_builder.build()` turns it into an SD prompt + a deterministic",
        "   pictographic collage (the img2img seed).",
        "3. `generator.generate_img2img()` runs SD-Turbo at 4 steps, strength 0.99.",
        "4. `pixelize()` enforces a true pixel-art look (64×64 grid, 16-color",
        "   palette, nearest-neighbor upscale).",
        "",
        "| seed | summary | subjects in prompt | collage | SD-Turbo doodle | pixel-perfect |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        subjects = "; ".join(r["subjects"])
        # Markdown table cells can't have raw newlines; escape pipes.
        subjects = subjects.replace("|", "\\|")
        summary = r["summary"].replace("|", "\\|")
        lines.append(
            f"| {r['seed']} | {summary} | {subjects} "
            f"| ![]({r['collage']}) | ![]({r['doodle']}) | ![]({r['pixel']}) |"
        )
    (SAMPLES / "random_weeks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    seeds = [11, 22, 33, 44]
    rows = []
    for s in seeds:
        print(f"\n[random] === seed={s} ===")
        row = run_one(s)
        print(f"[random] subjects: {row['subjects']}")
        rows.append(row)
    write_index(rows)
    print(f"\n[random] index -> samples/random_weeks.md")


if __name__ == "__main__":
    main()
