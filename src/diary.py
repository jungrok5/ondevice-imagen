"""Load and summarize one week of diary entries."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class WeekSummary:
    week_start: str
    week_end: str
    total_water_ml: int
    avg_water_ml_per_day: float
    water_event_count: int
    weather_counts: dict
    dominant_weather: str
    avg_temp_c: float
    place_categories: dict
    top_places: list
    daily_highlights: list


def load_week(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(week: dict) -> WeekSummary:
    entries = week.get("entries", [])
    water_total = 0
    water_count = 0
    weathers: Counter = Counter()
    temps: list[float] = []
    place_cats: Counter = Counter()
    place_names: Counter = Counter()
    highlights = []

    for day in entries:
        intakes = day.get("water_intake", [])
        day_water = sum(int(i.get("ml", 0)) for i in intakes)
        water_total += day_water
        water_count += len(intakes)

        weather = day.get("weather", {}) or {}
        if weather.get("summary"):
            weathers[weather["summary"]] += 1
        if isinstance(weather.get("temp_c"), (int, float)):
            temps.append(float(weather["temp_c"]))

        places = day.get("places", []) or []
        for p in places:
            if p.get("category"):
                place_cats[p["category"]] += 1
            if p.get("name"):
                place_names[p["name"]] += 1

        highlights.append({
            "date": day.get("date"),
            "weather": weather.get("summary"),
            "water_ml": day_water,
            "places": [p.get("name") for p in places if p.get("name")],
        })

    days = max(len(entries), 1)
    return WeekSummary(
        week_start=week.get("week_start", ""),
        week_end=week.get("week_end", ""),
        total_water_ml=water_total,
        avg_water_ml_per_day=round(water_total / days, 1),
        water_event_count=water_count,
        weather_counts=dict(weathers),
        dominant_weather=(weathers.most_common(1)[0][0] if weathers else "unknown"),
        avg_temp_c=round(sum(temps) / len(temps), 1) if temps else 0.0,
        place_categories=dict(place_cats),
        top_places=[name for name, _ in place_names.most_common(5)],
        daily_highlights=highlights,
    )


def summary_to_text(s: WeekSummary) -> str:
    """Human-readable summary string for debug or LLM hand-off."""
    parts = [
        f"Week {s.week_start} ~ {s.week_end}",
        f"Total water: {s.total_water_ml} ml across {s.water_event_count} sips",
        f"Avg/day: {s.avg_water_ml_per_day} ml",
        f"Dominant weather: {s.dominant_weather} (avg {s.avg_temp_c} C)",
        f"Place categories: {s.place_categories}",
        f"Top places: {', '.join(s.top_places) if s.top_places else 'none'}",
    ]
    return " | ".join(parts)
