"""Translate a single moment event (the data captured the moment the
user presses the button) into an SDXL prompt string.

Input shape:
  {
    "time":    "19:00",                      # 24-hour HH:MM
    "weather": "맑음",                        # Korean or English
    "date":    "2026-12-25",                 # YYYY-MM-DD
    "country": "대한민국",                     # Korean / English country name
    "city":    "서울시",                       # optional
    "place":   "무궁화 아파트",                 # POI string from reverse geocode
  }

Each field is mapped to a short English prompt phrase via small
dictionaries. The transformation is *deterministic* and *inspectable*
so the user can read the README table and see exactly which field
turned into which phrase.

Final prompt structure:
  {LORA_TRIGGER}, {STYLE_TAGS}, {translated event phrases}
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


LORA_TRIGGER = "DD-wte artstyle, worst-im-ever cartoon doodle"

STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, naive crude drawing"
)

NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)


# ---------- field → phrase tables (small, transparent, editable) ----------

def _time_phrase(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning, soft golden light"
    if 11 <= hour < 14:
        return "midday, bright"
    if 14 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 19:
        return "early evening, golden hour"
    if 19 <= hour < 22:
        return "evening dusk, lamps lit"
    return "night, dark windows, lamp glow"


def _season_phrase(month: int) -> str:
    if month in (12, 1, 2):
        return "winter, bare branches"
    if month in (3, 4, 5):
        return "spring, fresh leaves"
    if month in (6, 7, 8):
        return "summer, full green leaves"
    return "autumn, fallen leaves"


def _special_date_phrase(month: int, day: int) -> str | None:
    if (month, day) == (12, 25):
        return "Christmas day, fairy lights, festive"
    if (month, day) == (12, 31):
        return "New Year's eve, fireworks"
    if (month, day) == (1, 1):
        return "New Year's day, fresh start"
    if (month, day) == (2, 14):
        return "Valentine's, hearts"
    return None


_WEATHER = {
    # Korean
    "맑음": "clear sky",
    "흐림": "overcast sky",
    "구름": "overcast sky",
    "비": "raining, wet street",
    "눈": "snowing, snow on the ground",
    "바람": "windy, swirling leaves",
    # English
    "sunny": "clear sky",
    "clear": "clear sky",
    "cloudy": "overcast sky",
    "rainy": "raining, wet street",
    "snowy": "snowing, snow on the ground",
    "windy": "windy, swirling leaves",
}


_COUNTRY = {
    "대한민국": "Korean",
    "한국": "Korean",
    "korea": "Korean",
    "south korea": "Korean",
    "일본": "Japanese",
    "japan": "Japanese",
    "미국": "American",
    "usa": "American",
    "united states": "American",
}


# Substring matches against the place string. Multiple may apply
# (e.g. "광교포레스트 아파트" matches both "포레스트" and "아파트").
_POI_KEYWORDS = [
    ("아파트", "tall apartment buildings"),
    ("apartment", "tall apartment buildings"),
    ("포레스트", "forested area, tall pines"),
    ("forest", "forested area, tall pines"),
    ("공원", "park with trees"),
    ("park", "park with trees"),
    ("카페", "cozy cafe"),
    ("cafe", "cozy cafe"),
    ("역", "train station"),
    ("station", "train station"),
    ("광장", "city plaza"),
    ("plaza", "city plaza"),
    ("강", "riverside"),
    ("river", "riverside"),
    ("산", "mountain in the distance"),
    ("mountain", "mountain in the distance"),
    ("바다", "seaside"),
    ("beach", "seaside"),
    ("학교", "school building"),
    ("school", "school building"),
    ("도서관", "library"),
    ("library", "library"),
    ("시장", "outdoor market stalls"),
    ("market", "outdoor market stalls"),
]


# ---------- main builder ----------

@dataclass
class EventPrompt:
    positive: str
    negative: str
    breakdown: list[tuple[str, str]]  # [(input_field, output_phrase), ...]


def build_event_prompt(event: dict) -> EventPrompt:
    """Map a structured single-moment event to an SD prompt.

    Returns positive prompt, negative prompt, and a (field -> phrase)
    breakdown so README / UI can show which input became which phrase.
    """
    breakdown: list[tuple[str, str]] = []

    # 1. Time → time-of-day phrase
    if t := event.get("time"):
        try:
            h = int(str(t).split(":")[0])
            phrase = _time_phrase(h)
            breakdown.append((f"time={t}", phrase))
        except ValueError:
            phrase = ""
    else:
        phrase = ""
    if phrase:
        time_phrase_out = phrase
    else:
        time_phrase_out = None

    # 2. Date → season + special-date phrase
    season_phrase_out = None
    special_phrase_out = None
    if d := event.get("date"):
        try:
            dt = _dt.datetime.strptime(d, "%Y-%m-%d")
            season_phrase_out = _season_phrase(dt.month)
            breakdown.append((f"date={d} (season)", season_phrase_out))
            special = _special_date_phrase(dt.month, dt.day)
            if special:
                special_phrase_out = special
                breakdown.append((f"date={d} (special)", special))
        except ValueError:
            pass

    # 3. Weather → phrase
    weather_phrase_out = None
    if w := event.get("weather"):
        key = str(w).strip().lower()
        # Try Korean first (key not lowercased for Korean)
        weather_phrase_out = _WEATHER.get(str(w).strip()) or _WEATHER.get(key)
        if weather_phrase_out:
            breakdown.append((f"weather={w}", weather_phrase_out))

    # 4. Country → cultural anchor
    country_phrase_out = None
    if c := event.get("country"):
        key = str(c).strip().lower()
        anchor = _COUNTRY.get(str(c).strip()) or _COUNTRY.get(key)
        if anchor:
            country_phrase_out = f"{anchor} setting"
            breakdown.append((f"country={c}", country_phrase_out))

    # 5. POI place → keyword-matched phrases
    poi_phrases_out: list[str] = []
    if p := event.get("place"):
        for kw, phrase in _POI_KEYWORDS:
            if kw in p:
                poi_phrases_out.append(phrase)
                breakdown.append((f"place={p!r} matches '{kw}'", phrase))
        # No matches → just include the raw place name as a tail phrase
        if not poi_phrases_out:
            poi_phrases_out.append(f"a place called {p}")
            breakdown.append((f"place={p!r} (no keyword match)", poi_phrases_out[-1]))

    fragments: list[str] = []
    for f in (
        time_phrase_out,
        season_phrase_out,
        special_phrase_out,
        weather_phrase_out,
        country_phrase_out,
        *poi_phrases_out,
    ):
        if f:
            fragments.append(f)

    moment_block = ", ".join(fragments)
    positive = f"{LORA_TRIGGER}, {STYLE_TAGS}, {moment_block}"
    return EventPrompt(positive=positive, negative=NEGATIVE, breakdown=breakdown)
