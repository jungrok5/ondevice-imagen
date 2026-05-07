"""Dump the prompt that prompt_builder produces for each random week.

Quick way to see what the protagonist-rotation logic does without paying
for a full SD inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder  # noqa: E402
from scripts.random_weeks_doodle import make_random_week  # noqa: E402


def main() -> None:
    for seed in [11, 22, 33, 44]:
        week = make_random_week(seed)
        s = diary.summarize(week)
        built = prompt_builder.build(s, seed=seed)
        print(f"\n=== seed={seed} ===")
        print(f"  cups          : {s.water_event_count}")
        print(f"  weather       : {s.dominant_weather}  counts={s.weather_counts}")
        print(f"  places        : {s.place_categories}")
        print(f"  protagonist   : {built.protagonist}")
        print(f"  scores        : {built.debug_summary}")
        for i, m in enumerate(built.subjects):
            print(f"    moment {i+1}: {m}")
        print(f"  full prompt   : {built.positive}")


if __name__ == "__main__":
    main()
