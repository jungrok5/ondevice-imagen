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
    # IMPORTANT: time phrases describe time-of-day signals (light direction,
    # shadow length, lamp state, activity level) and intentionally AVOID
    # sky-color or cloud descriptors — those belong to weather phrases and
    # would conflict (e.g. "blue sky" + "rain falling" produces a confused
    # render). Night is the one exception: an unlit night sky IS the time
    # signal, so it stays.
    "early_morning": [
        "early morning, long horizontal shadows, very few people",
        "first light of day, soft side light, empty quiet street",
        "dawn, faint warm light, almost no traffic",
        "early morning hour, dim diffuse light, lone jogger",
    ],
    "morning": [
        "morning, soft slanted light, sharp small shadows",
        "mid-morning hour, school kids, busy commute",
        "morning light, crisp clean atmosphere, fresh start of the day",
        "morning, dewdrops on grass, golden side light",
    ],
    "midday": [
        "midday, sun directly overhead, very short shadows",
        "noon, lunchtime crowd, busy storefronts",
        "midday hour, bleaching overhead light, people on a quick lunch break",
        "noon, vendor stalls in full swing, vivid daylight",
    ],
    "afternoon": [
        "afternoon, slanted warm light, lengthening shadows",
        "mid-afternoon, lazy quiet hour, parents with kids on the street",
        "late afternoon, long shadows, golden side light",
        "afternoon, schoolchildren walking home, warm tones",
    ],
    "early_evening": [
        "early evening, golden hour, very long shadows",
        "sunset hour, warm orange light bouncing off walls",
        "dusk approaching, last warm light of the day, lamps starting to flicker on",
        "evening rush, returning commuters, sky darkening at the edges",
    ],
    "evening": [
        "evening, lamps lit, glowing windows, deep navy at the horizon",
        "early night, street lamps on, lit shop signs, fewer people",
        "twilight, first stars appearing, warm lamp glow spilling onto pavement",
        "evening, neon signs starting to dominate, dark sky overhead",
    ],
    "night": [
        "dark night sky, big bright moon, stars, deep navy blue",
        "late night, glowing yellow windows, street lamps casting pools of light, dark sky",
        "midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow",
        "deep night, moonlight on rooftops, dark windows, occasional lit lamp",
        "night, full moon casting silver light, very few lit windows, empty street",
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
        "wintertime, dry brown grass, frosted edges",
        "winter, people in puffy jackets and scarves, frozen puddles",
        "midwinter, evergreen pines holding patches of snow, bare deciduous trees",
    ],
    "spring": [
        "spring, fresh green leaves, cherry blossoms in bloom",
        "early spring, pink and white blossoms on trees, light layered clothing",
        "spring, new buds on branches, scattered petals on the ground",
        "late spring, full green canopy starting to fill in, magnolia blossoms",
        "springtime, daffodils and tulips along the edges, mild breeze ruffling petals",
    ],
    "summer": [
        "summer, lush full green leaves, intense green",
        "high summer, dense tree canopy, cicadas implied, t-shirts and shorts",
        "summertime, vivid green grass, deep green leaves",
        "midsummer, sunflowers and morning glories, heat shimmer in the air",
        "summer, kids in sandals, ice cream stand visible in the corner",
    ],
    "autumn": [
        "autumn, red and orange and yellow fallen leaves on the ground",
        "late autumn, bare-ish branches, piles of orange leaves",
        "autumn, maple leaves crimson, ginkgo leaves yellow on the ground",
        "early autumn, leaves just starting to turn yellow, mild crisp atmosphere",
        "autumn, persimmons hanging from a tree, light cardigans on people",
    ],
}


def _season_bucket(month: int) -> str:
    if month in (12, 1, 2):  return "winter"
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    return "autumn"


_SPECIAL_DATE_PHRASES: dict[tuple[int, int], list[str]] = {
    # solar-calendar dates only — lunar holidays (chuseok, seollal) shift
    # year-to-year and are not encoded here.
    (12, 24): [
        "Christmas eve, families gathering, candles in windows",
        "Christmas eve, snow gently falling, decorated shop fronts",
    ],
    (12, 25): [
        "Christmas day, fairy lights twinkling, Christmas tree visible",
        "Christmas, red-and-green wreath, festive ornaments",
        "Christmas, snowflakes falling, lit Christmas tree, Santa hat",
        "Christmas, gift boxes wrapped in red and gold ribbon, mistletoe",
    ],
    (12, 31): [
        "New Year's eve, fireworks bursting in the sky, sparklers",
        "New Year's eve, countdown banner, exploding fireworks above",
        "year-end festivity, neon countdown clock, crowds bundled in coats",
    ],
    (1, 1): [
        "New Year's day, sunrise on a fresh year, traditional banner",
        "New Year's day, pristine snow, '새해' calligraphy banner",
        "first day of the year, families in hanbok, calm festive mood",
    ],
    (2, 14): [
        "Valentine's day, pink and red hearts floating in the air",
        "Valentine's, heart shapes, roses, pink atmosphere",
        "Valentine's day, chocolate boxes in shop windows, couples walking",
    ],
    (3, 1): [
        "March First independence movement day, taegukgi flags hanging from windows",
    ],
    (5, 5): [
        "Children's day, colorful balloons, families with kids in the park",
        "어린이날, festive park scene, balloons and ice cream",
    ],
    (5, 8): [
        "Parents' day, children handing red carnations, warm family moment",
    ],
    (8, 15): [
        "Liberation day, taegukgi flags flying from balconies, festive mood",
        "광복절, Korean flags lining the street",
    ],
    (10, 9): [
        "Hangul day, calligraphy banners, '한글' brushwork",
    ],
    (10, 31): [
        "Halloween, jack-o-lanterns on doorsteps, kids in costumes",
        "Halloween night, cobwebs and pumpkins in shop windows",
    ],
    (11, 11): [
        "Pepero day, stacks of long thin biscuit boxes in store windows",
    ],
}


_WEATHER_PHRASES: dict[str, list[str]] = {
    "clear": [
        "clear blue sky, bright sunshine, no clouds",
        "perfectly clear sky, single fluffy cloud, sharp colors",
        "crisp clear weather, deep blue sky, vivid daylight",
        "sunny day, big white puffy clouds drifting, vivid contrast",
        "open clear sky, glittering reflections off windows, bright atmosphere",
    ],
    "overcast": [
        "overcast gray sky, flat diffuse light, no shadows",
        "cloudy sky, thick gray clouds, moody atmosphere",
        "completely cloudy, dull gray sky, hazy air",
        "heavy gray cloud cover, muted dull palette, soft uniform light",
        "thick cloud blanket, low ceiling, stillness in the air",
    ],
    "rain": [
        "rain falling in visible streaks, wet shiny pavement, puddles reflecting light",
        "heavy rain, raindrops splashing in puddles, dark wet streets",
        "raining steadily, wet umbrellas, water dripping from edges",
        "drizzle, fine rain in the air, soaked sidewalks",
        "rainstorm, sheets of rain at an angle, blurred distant shapes",
        "light rain, glistening leaves, rainbow puddle reflections",
    ],
    "snow": [
        "snowing heavily, big white snowflakes falling, snow piled on ground",
        "snow falling softly, white blanket on every surface, footprints in snow",
        "snowstorm, swirling snowflakes, pristine white snow drifts",
        "fluffy snow drifting down, snow hats on every surface, kids making snowman",
        "fresh snowfall, untouched white powder, sparkling crystals catching the light",
    ],
    "wind": [
        "windy, leaves swirling in the air, hair and clothes blown sideways",
        "strong wind, branches bending, papers flying",
        "gusty wind, plastic bag tumbling down the street, banners flapping",
        "blustery weather, hats clutched in hands, scarves trailing horizontally",
    ],
    "fog": [
        "thick fog, distant shapes blurred, low visibility",
        "foggy morning, mist hugging the ground, ghostly silhouettes",
        "dense fog, halos around every street lamp, muffled atmosphere",
        "thin mist hanging in the air, soft pastel washed colors",
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
