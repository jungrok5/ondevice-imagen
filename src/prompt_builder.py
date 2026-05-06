"""Build a Stable Diffusion prompt from a WeekSummary.

The user's intent (Korean): redraw clumsily, like a child scribbling in old MS Paint
with a mouse — recognizable but off, low-res, pixelated, deliberately bad.

We translate that intent into SD-friendly English tags + lift visual subjects from
the week's data so the postcard hints at what actually happened.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.diary import WeekSummary


# The user's signature style — kept identical run-to-run so the aesthetic is stable.
STYLE_TAGS = (
    "crude childlike doodle, scribbled with a mouse in old MS Paint, "
    "white background, lo-fi, pixelated, low-resolution, jagged lines, "
    "uneven shaky strokes, recognizable but distorted, awkward proportions, "
    "deliberately badly drawn, jpeg artifacts, naive art, postcard layout"
)

NEGATIVE_PROMPT = (
    "professional, polished, detailed, photorealistic, smooth, anti-aliased, "
    "high quality, beautiful, masterpiece, 4k, hd, sharp focus"
)


WEATHER_TOKENS = {
    "sunny": "smiling sun in the corner, scribbled rays",
    "cloudy": "lumpy clouds doodled across the top",
    "rainy": "stick-figure rain drops, wavy puddles",
    "windy": "squiggly wind lines, leaves flying sideways",
    "snowy": "scattered snowflakes drawn as asterisks",
}


CATEGORY_TOKENS = {
    "cafe": "wobbly coffee cup with steam squiggles",
    "park": "lopsided trees, crooked grass strokes",
    "work": "ugly rectangle building with mismatched windows",
    "bookstore": "stack of crooked books",
    "restaurant": "bowl with wavy noodles",
    "landmark": "tower drawn with shaky vertical lines",
    "market": "shopping cart with random shapes inside",
    "home": "tiny house with a triangle roof",
}


@dataclass
class BuiltPrompt:
    positive: str
    negative: str
    subjects: list[str]
    debug_summary: str


def build(summary: WeekSummary) -> BuiltPrompt:
    subjects: list[str] = []

    # Water as the protagonist — count drives how many cups
    cups = max(1, min(summary.water_event_count, 8))
    subjects.append(f"{cups} mismatched water cups arranged in a row")

    # Weather → mood element
    weather_tok = WEATHER_TOKENS.get(summary.dominant_weather)
    if weather_tok:
        subjects.append(weather_tok)

    # Top place categories → little vignettes
    seen = set()
    for cat in summary.place_categories.keys():
        tok = CATEGORY_TOKENS.get(cat)
        if tok and tok not in seen:
            subjects.append(tok)
            seen.add(tok)
        if len(subjects) >= 6:
            break

    subject_block = ", ".join(subjects)
    title_block = f"a one-week diary postcard, {subject_block}"

    positive = f"{title_block}, {STYLE_TAGS}"
    return BuiltPrompt(
        positive=positive,
        negative=NEGATIVE_PROMPT,
        subjects=subjects,
        debug_summary=(
            f"cups={cups} weather={summary.dominant_weather} "
            f"categories={list(summary.place_categories.keys())}"
        ),
    )
