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
# CLIP truncates at 77 tokens; the *front* of the prompt has the most influence.
# So this block is short and punchy — no redundant adjectives.
STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, jagged shaky lines, naive crude drawing"
)

NEGATIVE_PROMPT = (
    "color, photorealistic, polished, smooth, hd, high quality, sharp, beautiful, "
    "professional artwork, anti-aliased, gradient, realistic, painting"
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

    # Style FIRST — CLIP truncates at 77 tokens, and the front of the prompt
    # carries the most weight. Subjects come after so they can be trimmed.
    subject_block = ", ".join(subjects)
    positive = f"{STYLE_TAGS}, weekly diary postcard with {subject_block}"
    return BuiltPrompt(
        positive=positive,
        negative=NEGATIVE_PROMPT,
        subjects=subjects,
        debug_summary=(
            f"cups={cups} weather={summary.dominant_weather} "
            f"categories={list(summary.place_categories.keys())}"
        ),
    )
