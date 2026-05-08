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

Each field maps to a *pool* of short English phrases. The builder
picks one phrase per field per call via an internal RNG, so the same
event called twice can produce two different prompts (and two
different images downstream). Pass prompt_seed=<int> to make the
pick deterministic for tests.

Final prompt structure (per visual_style):
  {LORA_TRIGGERS[visual_style]}, {STYLE_TAGS?}, {translated event phrases}

The LoRA trigger is *required* for any style LoRA to actually take
effect (verified by sanity_lora.py — fusing without the trigger in
the prompt produces ~no change). The registry below maps a short
style name to the exact trigger phrase from each LoRA's card.
"""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass


# LoRA trigger registry. Keys are short style names; the user picks
# one of these for visual_style and the builder auto-prepends the
# trigger phrase. Triggers come from each LoRA card on Civitai/HF —
# do not edit unless the LoRA is replaced.
LORA_TRIGGERS: dict[str, str] = {
    "none":               "",
    "worstimever":        "DD-wte artstyle, worst-im-ever cartoon doodle",
    "mspaint_portraits":  "MSPaint drawing of",
    "lah_cute_social":    "cute doodle,",
    "pixel_art_xl":       "pixel art,",
    # Speed LoRAs (Lightning, Hyper-SD) take no trigger — they modify
    # the denoising schedule, not visual style. Don't add them here.
}

# Default style for new generations. Override via visual_style arg.
DEFAULT_VISUAL_STYLE = "worstimever"

STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, naive crude drawing"
)

NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)


# ---------- field → phrase pools (lists, picked via rng.choice) ----------
# Each phrase is a small English fragment; the builder joins them with
# commas. Make each phrase visually concrete — vague phrases like "dark"
# wash out under the doodle style; phrases naming explicit objects
# ("moon", "rain puddles", "snowflakes") survive better.

_TIME_PHRASES: dict[str, list[str]] = {
    "early_morning": [
        "early morning, pale dawn light, long shadows",
        "sunrise glow, pink-orange sky, quiet street",
        "first light of day, mist hovering low",
    ],
    "morning": [
        "morning, soft golden light, crisp air",
        "bright morning sunlight, blue sky",
        "late morning, warm sunshine, sharp shadows",
    ],
    "midday": [
        "midday sun directly overhead, bright bleaching light",
        "noon, harsh bright light, short shadows",
        "lunchtime, sun high, vivid colors",
    ],
    "afternoon": [
        "afternoon, slanted sunlight, warm tones",
        "mid-afternoon, soft shadows, calm light",
        "late afternoon, golden warm glow",
    ],
    "early_evening": [
        "early evening, golden hour, orange-pink sky",
        "sunset glow, red-purple clouds, long shadows",
        "dusk approaching, sun low on horizon, warm sky",
    ],
    "evening": [
        "evening dusk, lamps lit, deep blue sky",
        "early night, glowing windows, street lamps on",
        "twilight, first stars appearing, lamp glow on streets",
    ],
    "night": [
        "dark night sky, big bright moon, stars, deep navy blue",
        "late night, glowing yellow windows, street lamps casting pools of light, dark sky",
        "midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow",
        "deep night, moonlight on rooftops, dark windows, occasional lit lamp",
    ],
}


def _time_bucket(hour: int) -> str:
    if 5 <= hour < 8:    return "early_morning"
    if 8 <= hour < 11:   return "morning"
    if 11 <= hour < 14:  return "midday"
    if 14 <= hour < 17:  return "afternoon"
    if 17 <= hour < 19:  return "early_evening"
    if 19 <= hour < 22:  return "evening"
    return "night"


_SEASON_PHRASES: dict[str, list[str]] = {
    "winter": [
        "winter, bare branches, cold air visible as breath",
        "deep winter, leafless trees, icy ground",
        "wintertime, dry brown grass, gray cold sky",
    ],
    "spring": [
        "spring, fresh green leaves, cherry blossoms in bloom",
        "early spring, pink and white blossoms on trees, soft breeze",
        "spring, new buds on branches, scattered petals on ground",
    ],
    "summer": [
        "summer, lush full green leaves, intense green",
        "high summer, dense tree canopy, cicadas implied",
        "summertime, vivid green grass, deep green leaves",
    ],
    "autumn": [
        "autumn, red and orange and yellow fallen leaves on ground",
        "late autumn, bare-ish branches, piles of orange leaves",
        "autumn, maple leaves crimson, ginkgo leaves yellow on the ground",
    ],
}


def _season_bucket(month: int) -> str:
    if month in (12, 1, 2):  return "winter"
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    return "autumn"


_SPECIAL_DATE_PHRASES: dict[tuple[int, int], list[str]] = {
    (12, 25): [
        "Christmas day, fairy lights twinkling, Christmas tree visible",
        "Christmas, red-and-green wreath, festive ornaments",
        "Christmas, snowflakes falling, lit Christmas tree, Santa hat",
    ],
    (12, 31): [
        "New Year's eve, fireworks bursting in the sky, sparklers",
        "New Year's eve, countdown banner, exploding fireworks above",
    ],
    (1, 1): [
        "New Year's day, sunrise on a fresh year, traditional banner",
        "New Year's day, pristine snow, '새해' calligraphy banner",
    ],
    (2, 14): [
        "Valentine's day, pink and red hearts floating in the air",
        "Valentine's, heart shapes, roses, pink atmosphere",
    ],
}


_WEATHER_PHRASES: dict[str, list[str]] = {
    "clear": [
        "clear blue sky, bright sunshine, no clouds",
        "perfectly clear sky, sun visible, sharp shadows",
        "crisp clear weather, deep blue sky, single fluffy cloud",
    ],
    "overcast": [
        "overcast gray sky, flat diffuse light, no shadows",
        "cloudy sky, thick gray clouds, moody atmosphere",
        "completely cloudy, dull gray sky, hazy air",
    ],
    "rain": [
        "rain falling in visible streaks, wet shiny pavement, puddles reflecting light",
        "heavy rain, raindrops splashing in puddles, dark wet streets",
        "raining steadily, wet umbrellas, water dripping from edges",
        "drizzle, fine rain in the air, soaked sidewalks",
    ],
    "snow": [
        "snowing heavily, big white snowflakes falling, snow piled on ground",
        "snow falling softly, white blanket on every surface, footprints in snow",
        "snowstorm, swirling snowflakes, pristine white snow drifts",
    ],
    "wind": [
        "windy, leaves swirling in the air, hair and clothes blown sideways",
        "strong wind, branches bending, papers flying",
    ],
    "fog": [
        "thick fog, distant shapes blurred, low visibility",
        "foggy morning, mist hugging the ground, ghostly silhouettes",
    ],
}

# Korean / English raw weather term → bucket key
_WEATHER_LOOKUP: dict[str, str] = {
    # Korean
    "맑음": "clear", "맑": "clear",
    "흐림": "overcast", "흐": "overcast", "구름": "overcast",
    "비": "rain", "소나기": "rain",
    "눈": "snow", "폭설": "snow",
    "바람": "wind",
    "안개": "fog",
    # English
    "sunny": "clear", "clear": "clear",
    "cloudy": "overcast", "overcast": "overcast",
    "rain": "rain", "rainy": "rain",
    "snow": "snow", "snowy": "snow",
    "wind": "wind", "windy": "wind",
    "fog": "fog", "foggy": "fog",
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


# Korean place-name → English (Revised Romanization or common English).
# Longer keys first so substring matching prefers e.g. "광장시장" over "광장".
_PLACE_PROPER_NOUNS: list[tuple[str, str]] = [
    # Markets / landmarks
    ("광장시장",   "Gwangjang Market"),
    ("남대문시장", "Namdaemun Market"),
    ("동대문시장", "Dongdaemun Market"),
    ("노량진시장", "Noryangjin Fish Market"),
    ("경동시장",   "Gyeongdong Market"),

    # Palaces / historic sites
    ("경복궁",     "Gyeongbokgung Palace"),
    ("창덕궁",     "Changdeokgung Palace"),
    ("덕수궁",     "Deoksugung Palace"),
    ("종묘",       "Jongmyo Shrine"),
    ("광화문",     "Gwanghwamun"),
    ("남산타워",   "Namsan Tower"),
    ("남산",       "Namsan Mountain"),
    ("북한산",     "Bukhansan Mountain"),
    ("관악산",     "Gwanaksan Mountain"),
    ("도봉산",     "Dobongsan Mountain"),
    ("청계천",     "Cheonggyecheon Stream"),

    # Rivers / parks
    ("한강공원",   "Hangang Park by the river"),
    ("한강",       "Hangang River"),
    ("올림픽공원", "Olympic Park"),
    ("서울숲",     "Seoul Forest park"),
    ("월드컵공원", "World Cup Park"),

    # Districts / neighborhoods
    ("강남역",     "Gangnam Station area"),
    ("강남",       "Gangnam district"),
    ("홍대입구",   "Hongdae entrance area"),
    ("홍대",       "Hongdae nightlife district"),
    ("이태원",     "Itaewon district"),
    ("명동",       "Myeongdong shopping street"),
    ("종로",       "Jongno old downtown"),
    ("을지로",     "Euljiro retro district"),
    ("성수",       "Seongsu cafe district"),
    ("연남동",     "Yeonnam-dong"),
    ("망원",       "Mangwon"),
    ("합정",       "Hapjeong"),
    ("여의도",     "Yeouido"),
    ("잠실",       "Jamsil"),
    ("압구정",     "Apgujeong"),
    ("청담",       "Cheongdam"),
    ("삼청동",     "Samcheong-dong traditional alleys"),
    ("북촌",       "Bukchon Hanok village"),
    ("서촌",       "Seochon village"),
    ("인사동",     "Insadong cultural street"),

    # Cities (broad)
    ("서울시",     "Seoul"),
    ("서울",       "Seoul"),
    ("부산",       "Busan seaside city"),
    ("제주",       "Jeju Island"),
    ("경주",       "Gyeongju historic city"),
    ("인천",       "Incheon port city"),
    ("대구",       "Daegu"),
    ("대전",       "Daejeon"),
    ("광주",       "Gwangju"),
    ("수원",       "Suwon"),

    # Generic landmarks at the tail (so they only match if no specific
    # name matched first)
    ("아파트",     "tall Korean apartment buildings"),
    ("apartment",  "tall apartment buildings"),
    ("포레스트",   "forested area, tall pines"),
    ("forest",     "forested area, tall pines"),
    ("공원",       "park with trees and benches"),
    ("park",       "park with trees and benches"),
    ("카페",       "cozy Korean cafe"),
    ("cafe",       "cozy cafe"),
    ("역",         "subway station entrance"),
    ("station",    "train station"),
    ("광장",       "city plaza, open square"),
    ("plaza",      "city plaza"),
    ("강",         "riverside"),
    ("river",      "riverside"),
    ("산",         "mountain in the distance"),
    ("mountain",   "mountain in the distance"),
    ("바다",       "seaside, ocean horizon"),
    ("beach",      "sandy beach, ocean"),
    ("학교",       "school building, schoolyard"),
    ("school",     "school building"),
    ("도서관",     "quiet library, bookshelves"),
    ("library",    "library"),
    ("시장",       "outdoor Korean market stalls, food vendors"),
    ("market",     "outdoor market stalls"),
    ("편의점",     "Korean convenience store, lit signs"),
    ("convenience","convenience store"),
    ("백화점",     "department store"),
    ("학원",       "after-school academy building"),
    ("교회",       "church with cross steeple"),
    ("절",         "Buddhist temple, traditional roof"),
    ("temple",     "Buddhist temple, traditional roof"),
]


# ---------- main builder ----------

@dataclass
class EventPrompt:
    positive: str
    negative: str
    breakdown: list[tuple[str, str]]  # [(input_field, output_phrase), ...]


def _translate_place(place: str, rng: random.Random) -> list[tuple[str, str]]:
    """Return list of (matched_keyword, english_phrase) pairs.

    Walks _PLACE_PROPER_NOUNS in order (longest specific names first)
    and collects every match. The same English phrase is not added
    twice. If nothing matches, returns a single fallback that romanizes
    nothing — just embeds the raw place string.
    """
    out: list[tuple[str, str]] = []
    seen_phrases: set[str] = set()
    remaining = place
    for kw, phrase in _PLACE_PROPER_NOUNS:
        if kw in remaining and phrase not in seen_phrases:
            out.append((kw, phrase))
            seen_phrases.add(phrase)
            # consume the matched substring so a later shorter key
            # (e.g. "광장") does not double-match the same characters
            # already covered by a longer key (e.g. "광장시장").
            remaining = remaining.replace(kw, " ", 1)
    if not out:
        out.append(("", f"a place called {place}"))
    # Tiny shuffle so order varies between calls
    rng.shuffle(out)
    return out


def build_event_prompt(
    event: dict,
    visual_style: str = DEFAULT_VISUAL_STYLE,
    include_style_tags: bool = True,
    prompt_seed: int | None = None,
) -> EventPrompt:
    """Map a structured single-moment event to an SD prompt.

    visual_style       — which LoRA's trigger to auto-prepend. Must be
                         a key in LORA_TRIGGERS (see registry above).
                         Use "none" to skip the trigger entirely.
    include_style_tags — if True, prepend the deliberately-bad STYLE_TAGS
                         block. Set False to test data-only baseline.
    prompt_seed        — int to make phrase-pool picks deterministic.
                         None (default) = fresh randomness every call,
                         so same event can produce different prompts.

    Returns positive prompt, negative prompt, and a (field -> phrase)
    breakdown so README / UI can show which input became which phrase.
    """
    if visual_style not in LORA_TRIGGERS:
        raise ValueError(
            f"unknown visual_style {visual_style!r}; "
            f"choose one of {list(LORA_TRIGGERS)}"
        )
    trigger = LORA_TRIGGERS[visual_style]
    rng = random.Random(prompt_seed)

    breakdown: list[tuple[str, str]] = []

    # 1. Time → time-of-day phrase
    time_phrase_out: str | None = None
    if t := event.get("time"):
        try:
            h = int(str(t).split(":")[0])
            bucket = _time_bucket(h)
            time_phrase_out = rng.choice(_TIME_PHRASES[bucket])
            breakdown.append((f"time={t} ({bucket})", time_phrase_out))
        except ValueError:
            pass

    # 2. Date → season + special-date phrase
    season_phrase_out: str | None = None
    special_phrase_out: str | None = None
    if d := event.get("date"):
        try:
            dt = _dt.datetime.strptime(d, "%Y-%m-%d")
            sb = _season_bucket(dt.month)
            season_phrase_out = rng.choice(_SEASON_PHRASES[sb])
            breakdown.append((f"date={d} ({sb})", season_phrase_out))
            sp_pool = _SPECIAL_DATE_PHRASES.get((dt.month, dt.day))
            if sp_pool:
                special_phrase_out = rng.choice(sp_pool)
                breakdown.append((f"date={d} (special)", special_phrase_out))
        except ValueError:
            pass

    # 3. Weather → phrase
    weather_phrase_out: str | None = None
    if w := event.get("weather"):
        raw = str(w).strip()
        bucket = _WEATHER_LOOKUP.get(raw) or _WEATHER_LOOKUP.get(raw.lower())
        if bucket:
            weather_phrase_out = rng.choice(_WEATHER_PHRASES[bucket])
            breakdown.append((f"weather={w} ({bucket})", weather_phrase_out))

    # 4. Country → cultural anchor
    country_phrase_out: str | None = None
    if c := event.get("country"):
        anchor = _COUNTRY.get(str(c).strip()) or _COUNTRY.get(str(c).strip().lower())
        if anchor:
            country_phrase_out = f"{anchor} setting"
            breakdown.append((f"country={c}", country_phrase_out))

    # 5. Place → proper-noun romanization + POI keyword phrases
    poi_phrases_out: list[str] = []
    if p := event.get("place"):
        for matched_kw, phrase in _translate_place(p, rng):
            poi_phrases_out.append(phrase)
            tag = f"place={p!r} matches '{matched_kw}'" if matched_kw else f"place={p!r} (raw)"
            breakdown.append((tag, phrase))

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
    # Order matters because CLIP truncates at 77 tokens. Put the LoRA
    # trigger first (essential for activation), then the user-data
    # phrases (the whole point of the prompt), and only THEN the
    # generic STYLE_TAGS — so if anything gets truncated it's the
    # generic style hints, not the user's actual moment.
    parts: list[str] = []
    if trigger:
        parts.append(trigger.rstrip(",").rstrip())
    if moment_block:
        parts.append(moment_block)
    if include_style_tags:
        parts.append(STYLE_TAGS)
    positive = ", ".join(parts)
    return EventPrompt(positive=positive, negative=NEGATIVE, breakdown=breakdown)
