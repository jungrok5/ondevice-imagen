"""Build a Stable Diffusion prompt from a WeekSummary.

Per-week design: each week's data has different *distinctive* signals
(unusually many sips, a streak of bad weather, one place dominating, all
the sips happened at dawn, etc.). The builder detects which axis is
most distinctive *for this particular week* and frames the postcard
around it. That makes every result feel like it actually responds to
the data instead of always being a row of cups.

Four protagonist axes:
  - cups   : water-event count is unusually high or low
  - weather: a non-default weather (rainy / snowy / windy / cloudy)
             dominates the week
  - place  : one place category accounts for half-or-more of all visits
  - time   : sips heavily concentrated in one part of the day

When two axes tie, weighted-random pick by score keeps results varied
across runs even with identical data.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from src.diary import WeekSummary


# Style block stays the same — short and front-loaded.
STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, jagged shaky lines, naive crude drawing"
)

NEGATIVE_PROMPT = (
    "color, photorealistic, polished, smooth, hd, high quality, sharp, beautiful, "
    "professional artwork, anti-aliased, gradient, realistic, painting"
)


# Phrase pools — pick one per pool with rng.choice.
WEATHER_PHRASES = {
    "sunny": [
        "soft afternoon sunlight pooling on a wooden table",
        "morning sunbeams through a window",
        "a sunny park bench in late afternoon",
    ],
    "cloudy": [
        "a soft overcast sky over rooftops",
        "muted cloudy light through a window",
        "a quiet cloudy afternoon street",
    ],
    "rainy": [
        "rain streaks on a windowpane",
        "wet pavement reflecting streetlight",
        "umbrellas crossing a quiet alley",
    ],
    "windy": [
        "leaves swirling in afternoon air",
        "a curtain billowing in a quiet room",
        "wind-blown branches against a sky",
    ],
    "snowy": [
        "snow drifting past a window",
        "footprints in fresh morning snow",
        "a small snowy street under streetlamps",
    ],
}


PLACE_PHRASES = {
    "cafe": [
        "a corner cafe with steam rising from a teapot",
        "a cozy cafe window seat with a coffee cup",
        "a marble counter with a single espresso",
    ],
    "park": [
        "a wooden bench under tall trees",
        "a quiet path through autumn grass",
        "afternoon light filtering through leaves",
    ],
    "work": [
        "a desk by a window with a glass of water",
        "an empty office at lunch hour",
        "a notebook and pen on a quiet desk",
    ],
    "bookstore": [
        "rows of well-loved books on wooden shelves",
        "a small reading nook with a lamp",
        "an open book on a wooden table",
    ],
    "restaurant": [
        "a steaming bowl of noodles on a wooden table",
        "a small kitchen counter at dinnertime",
        "a quiet noodle shop at night",
    ],
    "landmark": [
        "a tower silhouette on the evening horizon",
        "a stone bridge in afternoon light",
        "a fountain in a public square",
    ],
    "market": [
        "a fruit stand under canvas awnings",
        "a small grocery basket on a counter",
        "produce in soft warm light",
    ],
    "home": [
        "a quiet kitchen at dawn",
        "a sun-warmed living room",
        "a bedroom with morning light through curtains",
    ],
}


TIME_PHRASES = {
    "morning": [
        "first light through curtains at dawn",
        "a quiet kitchen at sunrise",
        "morning steam from a fresh cup",
    ],
    "afternoon": [
        "soft midday hush",
        "afternoon light across a wooden floor",
        "a long shadow on a quiet street",
    ],
    "evening": [
        "warm evening light through a window",
        "a desk lamp glowing at dusk",
        "the last sun on a building wall",
    ],
    "night": [
        "a quiet kitchen at midnight",
        "lamplight on a desk after hours",
        "a darkened room with a single window",
    ],
}


# When *cups* is the protagonist, render water as the focus instead of as
# a count. Two phrasings: "a lot" and "barely any".
CUP_PHRASES_HIGH = [
    "a counter crowded with mismatched water cups",
    "many half-finished glasses on a wooden table",
    "a long row of small water cups",
]

CUP_PHRASES_LOW = [
    "a single forgotten glass of water",
    "a lone cup of water on a quiet table",
    "an empty glass beside an open window",
]


@dataclass
class BuiltPrompt:
    positive: str
    negative: str
    subjects: list[str]
    protagonist: str
    debug_summary: str


def _bucket_times(hours: list[int]) -> Counter:
    b: Counter = Counter()
    for h in hours:
        if 5 <= h < 11:
            b["morning"] += 1
        elif 11 <= h < 17:
            b["afternoon"] += 1
        elif 17 <= h < 22:
            b["evening"] += 1
        else:
            b["night"] += 1
    return b


def _score_axes(s: WeekSummary) -> dict[str, float]:
    """Return a non-negative distinctiveness score per axis.

    Weather scores are capped unless rainy/snowy dominates 6+ days, so
    a merely cloudy or windy majority does not drown out other axes.
    Place / time thresholds are 40% so a strong but not absolute lean
    can still win.
    """
    scores = {"cups": 0.0, "weather": 0.0, "place": 0.0, "time": 0.0}

    # Cups: extreme high (>= 25) or extreme low (<= 5)
    if s.water_event_count >= 25:
        scores["cups"] = 80.0 + (s.water_event_count - 25)
    elif s.water_event_count <= 5:
        scores["cups"] = 70.0

    # Weather: only a 6+ day streak of rainy/snowy is a strong protagonist;
    # everything else is capped so it does not drown other axes.
    weight = {"rainy": 22, "snowy": 22, "windy": 10, "cloudy": 8, "sunny": 4}
    days = s.weather_counts.get(s.dominant_weather, 0)
    raw = days * weight.get(s.dominant_weather, 0)
    if days >= 6 and s.dominant_weather in ("rainy", "snowy"):
        scores["weather"] = raw  # let the streak dominate
    else:
        scores["weather"] = min(raw, 55.0)

    # Place: one category accounts for >= 40% of visits
    total_visits = sum(s.place_categories.values())
    if total_visits > 0:
        top_count = max(s.place_categories.values())
        ratio = top_count / total_visits
        if ratio >= 0.4:
            scores["place"] = 60.0 + ratio * 50.0

    # Time: sips concentrated >= 40% in one bucket; morning/night more
    # distinctive than afternoon/evening
    bucket = _bucket_times(s.water_hours)
    total_b = sum(bucket.values())
    if total_b > 0:
        for kind, c in bucket.items():
            ratio = c / total_b
            if ratio >= 0.4:
                bonus = 35 if kind in ("morning", "night") else 18
                scores["time"] = max(scores["time"], bonus + ratio * 50.0)

    return scores


def _pick_protagonist(scores: dict[str, float], rng: random.Random) -> str:
    """Deterministic argmax — same data always produces the same
    protagonist axis. Variety comes from the data itself (different
    week → different distinctive axis), not from rerunning the same
    week. The rng is still used for *phrase* selection within the axis.
    """
    if not any(scores.values()):
        return "weather"
    return max(scores, key=lambda k: (scores[k], k))


def _pick_top_place_category(s: WeekSummary, rng: random.Random) -> Optional[str]:
    if not s.place_categories:
        return None
    # Sort by visit count desc; tie-break random for variety
    items = sorted(
        s.place_categories.items(), key=lambda kv: (-kv[1], rng.random())
    )
    return items[0][0]


def _pick_top_time_bucket(s: WeekSummary) -> Optional[str]:
    bucket = _bucket_times(s.water_hours)
    if not sum(bucket.values()):
        return None
    return bucket.most_common(1)[0][0]


def build(summary: WeekSummary, seed: Optional[int] = None) -> BuiltPrompt:
    rng = random.Random(seed)

    scores = _score_axes(summary)
    protagonist = _pick_protagonist(scores, rng)

    moments: list[str] = []

    # Lead with the protagonist's dramatic phrase
    if protagonist == "cups":
        if summary.water_event_count >= 25:
            moments.append(rng.choice(CUP_PHRASES_HIGH))
        else:
            moments.append(rng.choice(CUP_PHRASES_LOW))
    elif protagonist == "weather":
        moments.append(rng.choice(WEATHER_PHRASES[summary.dominant_weather]))
    elif protagonist == "place":
        cat = _pick_top_place_category(summary, rng)
        if cat and cat in PLACE_PHRASES:
            moments.append(rng.choice(PLACE_PHRASES[cat]))
    elif protagonist == "time":
        kind = _pick_top_time_bucket(summary)
        if kind:
            moments.append(rng.choice(TIME_PHRASES[kind]))

    # Add 2-3 supporting moments from the *other* axes (skip the
    # protagonist's axis to avoid redundancy)
    if protagonist != "weather":
        moments.append(rng.choice(WEATHER_PHRASES[summary.dominant_weather]))
    if protagonist != "place":
        cat = _pick_top_place_category(summary, rng)
        if cat and cat in PLACE_PHRASES:
            moments.append(rng.choice(PLACE_PHRASES[cat]))
    if protagonist != "time":
        kind = _pick_top_time_bucket(summary)
        if kind:
            moments.append(rng.choice(TIME_PHRASES[kind]))

    # Cap at 4 to stay under CLIP's 77-token budget once style block is
    # prepended.
    moments = [m for m in moments if m][:4]
    if not moments:
        moments.append("a quiet kitchen at dawn")

    subject_block = ", ".join(moments)
    positive = f"{STYLE_TAGS}, a postcard of one quiet week: {subject_block}"

    return BuiltPrompt(
        positive=positive,
        negative=NEGATIVE_PROMPT,
        subjects=moments,
        protagonist=protagonist,
        debug_summary=(
            f"protagonist={protagonist} scores={ {k: round(v, 1) for k, v in scores.items()} }"
        ),
    )
