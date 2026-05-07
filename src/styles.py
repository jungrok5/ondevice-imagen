"""Aesthetic presets for the postcard.

Each preset is a (positive_prefix, negative_extra) pair. The diary-derived
subject block is appended to positive_prefix at generation time.

Style tokens go FIRST so they survive CLIP's 77-token truncation. Each
preset is intentionally short and unambiguous — overloading with synonyms
dilutes rather than reinforces the look.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    name: str
    positive_prefix: str
    negative_extra: str


WATERCOLOR_DIARY = StylePreset(
    name="watercolor_diary",
    positive_prefix=(
        "watercolor painting on cold-press paper, soft pastel washes, "
        "loose hand-painted brush strokes, paper texture, gentle journal page, "
        "wet-on-wet bleeding edges"
    ),
    negative_extra=(
        "photorealistic, sharp focus, digital, vector, 3d render"
    ),
)


RISOGRAPH_ZINE = StylePreset(
    name="risograph_zine",
    positive_prefix=(
        "risograph print, two-color halftone, cobalt blue and warm red, "
        "grainy newsprint texture, indie zine illustration, "
        "limited palette, slight registration misalignment"
    ),
    negative_extra=(
        "smooth gradients, photorealistic, full color spectrum, glossy"
    ),
)


CRAYON_PICTUREBOOK = StylePreset(
    name="crayon_picturebook",
    positive_prefix=(
        "crayon and colored pencil drawing on textured beige paper, "
        "warm childlike picture book illustration, "
        "naive cozy hand-drawn lines, gentle storybook page"
    ),
    negative_extra=(
        "photorealistic, dark, edgy, digital, smooth, glossy"
    ),
)


INK_AND_WASH_JOURNAL = StylePreset(
    name="ink_and_wash_journal",
    positive_prefix=(
        "delicate pen and ink line drawing with light watercolor wash, "
        "traveler's sketchbook journal page, loose hand-drawn linework, "
        "white margins, handwritten feel"
    ),
    negative_extra=(
        "photorealistic, vector, 3d, glossy, full saturation"
    ),
)


MINHWA_HANJI = StylePreset(
    name="minhwa_hanji",
    positive_prefix=(
        "Korean minhwa folk painting on hanji rice paper, "
        "traditional ink and mineral pigment, flat decorative composition, "
        "muted earthy palette, folk art motifs"
    ),
    negative_extra=(
        "photorealistic, western, anime, manga, 3d, smooth"
    ),
)


GHIBLI_PAINTERLY = StylePreset(
    name="ghibli_painterly",
    positive_prefix=(
        "studio ghibli inspired painterly illustration, hand-drawn animation cel, "
        "warm golden-hour lighting, soft cumulus clouds, lush detailed background, "
        "nostalgic atmosphere"
    ),
    negative_extra=(
        "photorealistic, 3d render, harsh shadows, low quality"
    ),
)


PRESETS: dict[str, StylePreset] = {
    p.name: p
    for p in [
        WATERCOLOR_DIARY,
        RISOGRAPH_ZINE,
        CRAYON_PICTUREBOOK,
        INK_AND_WASH_JOURNAL,
        MINHWA_HANJI,
        GHIBLI_PAINTERLY,
    ]
}


BASE_NEGATIVE = (
    "lowres, jpeg artifacts, watermark, text, blurry, ugly, deformed, "
    "extra fingers, missing fingers, bad anatomy"
)


def build_prompt(preset: StylePreset, subject: str) -> tuple[str, str]:
    """Compose (positive, negative) prompts for a given preset + subject string."""
    positive = f"{preset.positive_prefix}, {subject}"
    negative = f"{BASE_NEGATIVE}, {preset.negative_extra}"
    return positive, negative
