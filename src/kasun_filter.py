"""한심함 (kasun) maximizer — push SD output toward the actual aesthetics
of the 하찮은 프롬프트 trend.

Pre-filter steps that meaningfully improve both modes:
  - MedianFilter to flatten gradients into solid regions
  - Saturation boost so quantize picks vivid colors instead of muddy grey

This module's parameters were tuned by comparing the output to the
heraldcorp Sam Altman reference doodle and the OpenAI-logo Instagram
post; both are saved under samples/reference/.

After looking at real reference outputs (the Sam Altman heraldcorp doodle
and the OpenAI-logo-only Instagram doodle) it's clear there are TWO
distinct sub-styles inside what people call "낙서풍":

  - **단색선 (line-only)**: black lines on white paper, no colour fill.
    Closest to the OpenAI-logo doodle on @openai's profile grid.
    Achieved by edge detection + threshold + dilation.

  - **컬러 낙서풍 (color flat-fill)**: 4-6 flat colours bounded by thick
    black outlines, like a kid filling in a colouring book. Closest to
    the Sam Altman heraldcorp doodle.
    Achieved by posterise + edge detection on the posterised image,
    composite back together.

Both pipelines are pure pixel ops — downsample, edge filter, threshold,
palette quantize, dilate, composite, nearest upscale. Each maps 1:1 to
iOS Core Image / Android Bitmap APIs. No extra ML model on top of
SD-Turbo.

The earlier bilevel-only `kasun()` is kept as `kasun_bilevel` for
reference — it gives a Game-Boy-bitmap look that is its own thing but
not what 하찮은 프롬프트 actually produces.
"""
from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Mode 1 — 단색선  (line-only, black on white)
# ---------------------------------------------------------------------------

def kasun_line(
    img: Image.Image,
    grid: int = 128,
    line_threshold: int = 110,
    dilate: int = 0,
    smooth: int = 5,
    output_size: int = 1024,
) -> Image.Image:
    """Edge-only black-on-white doodle (the OpenAI-logo aesthetic).

    grid:           edge length of the small intermediate image.
                    Smaller = simpler, more pathetic linework.
    line_threshold: 0..255 cutoff applied to FIND_EDGES output. Only
                    edges *stronger* than this survive. Higher = fewer
                    cleaner lines.
    dilate:         number of MinFilter passes to thicken black lines.
                    0 = thin marker, 1-2 = thicker brush. The
                    OpenAI-logo aesthetic is ~0.
    smooth:         MedianFilter kernel size before edge detection.
                    Higher kills SD high-frequency texture so only
                    region-boundary lines survive.
    output_size:    final upscaled PNG edge length.
    """
    src = img.convert("RGB")
    if smooth > 1:
        src = src.filter(ImageFilter.MedianFilter(smooth))

    small = src.resize((grid, grid), Image.LANCZOS)
    gray = small.convert("L")

    edges = gray.filter(ImageFilter.FIND_EDGES)
    bw = edges.point(lambda p: 0 if p > line_threshold else 255, mode="L")

    for _ in range(max(0, dilate)):
        bw = bw.filter(ImageFilter.MinFilter(3))

    return bw.convert("RGB").resize((output_size, output_size), Image.NEAREST)


# ---------------------------------------------------------------------------
# Mode 2 — 컬러 낙서풍  (flat colour fill + thick outlines)
# ---------------------------------------------------------------------------

def kasun_color(
    img: Image.Image,
    grid: int = 128,
    colors: int = 6,
    line_threshold: int = 60,
    dilate: int = 0,
    smooth: int = 5,
    saturation: float = 1.6,
    output_size: int = 1024,
) -> Image.Image:
    """Flat colour fill + thin black outline doodle (the Sam Altman
    heraldcorp aesthetic).

    grid:           edge length of the small intermediate image.
    colors:         palette size after median-cut. 5-6 is the sweet
                    spot for "bucket-fill" feel.
    line_threshold: cutoff for FIND_EDGES on the *posterised* image.
    dilate:         number of MaxFilter passes on the edge mask. 0 =
                    sharpie-thin lines (closer to the reference doodle).
    smooth:         MedianFilter kernel size before quantize. Flattens
                    SD's gradient noise into solid regions.
    saturation:     ImageEnhance.Color factor before quantize so the
                    palette picks vivid hues instead of muddy greys.
    output_size:    final upscaled PNG edge length.
    """
    src = img.convert("RGB")
    if smooth > 1:
        src = src.filter(ImageFilter.MedianFilter(smooth))
    if saturation != 1.0:
        src = ImageEnhance.Color(src).enhance(saturation)

    small = src.resize((grid, grid), Image.LANCZOS)

    # FASTOCTREE produces a more *uniform* palette across the colour
    # cube than median-cut, so a saturated input does not collapse to
    # one tone the way it does with median-cut on a white-dominant image.
    quantized = small.quantize(
        colors=colors, method=Image.Quantize.FASTOCTREE
    ).convert("RGB")

    edges = quantized.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mask = edges.point(lambda p: 255 if p > line_threshold else 0, mode="L")

    for _ in range(max(0, dilate)):
        edge_mask = edge_mask.filter(ImageFilter.MaxFilter(3))

    black = Image.new("RGB", quantized.size, (0, 0, 0))
    composite = Image.composite(black, quantized, edge_mask)

    return composite.resize((output_size, output_size), Image.NEAREST)


# ---------------------------------------------------------------------------
# Bonus — extreme bitmap, kept from v1 for the "Game Boy" aesthetic
# ---------------------------------------------------------------------------

GAMEBOY_PALETTE = [
    (15, 56, 15),
    (48, 98, 48),
    (139, 172, 15),
    (155, 188, 15),
]


def kasun_bilevel(
    img: Image.Image,
    grid: int = 64,
    threshold: int | None = None,
    output_size: int = 1024,
    auto_invert: bool = True,
) -> Image.Image:
    """1-bit pure black/white pixel art (Game Boy / vintage screen feel).

    Not actually what 하찮은 프롬프트 looks like — that aesthetic has
    flat colour or single-line drawings, not stark bitmap. Kept here as
    a separate retro mode.
    """
    src = img.convert("RGB")
    small = src.resize((grid, grid), Image.LANCZOS)

    gray = ImageOps.autocontrast(small.convert("L"))
    if auto_invert:
        mean = sum(gray.getdata()) / (gray.width * gray.height)
        if mean < 110:
            gray = ImageOps.invert(gray)

    thr = threshold if threshold is not None else _otsu_threshold(gray)
    bilevel = gray.point(lambda p: 255 if p > thr else 0, mode="1")
    return bilevel.convert("RGB").resize((output_size, output_size), Image.NEAREST)


def kasun_with_palette(
    img: Image.Image,
    grid: int = 64,
    palette: list[tuple[int, int, int]] | None = None,
    output_size: int = 1024,
) -> Image.Image:
    """Pixelize + remap to a fixed palette (e.g. Game Boy 4-color)."""
    if palette is None:
        palette = GAMEBOY_PALETTE
    src = img.convert("RGB")
    small = src.resize((grid, grid), Image.LANCZOS)
    quantized = small.quantize(
        palette=_palette_image(palette),
        dither=Image.Dither.NONE,
    ).convert("RGB")
    return quantized.resize((output_size, output_size), Image.NEAREST)


# ---------------------------------------------------------------------------
# Backward-compat alias (some scripts still call kasun() at import time).
# ---------------------------------------------------------------------------

def kasun(
    img: Image.Image,
    grid: int = 96,
    colors: int = 5,
    output_size: int = 1024,
    **_unused,
) -> Image.Image:
    """Default 'kasun' = the colour flat-fill doodle. Use kasun_line()
    for the pure single-colour-line variant.
    """
    return kasun_color(
        img, grid=grid, colors=colors, output_size=output_size
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _otsu_threshold(gray: Image.Image) -> int:
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 128
    sum_total = sum(i * h for i, h in enumerate(hist))
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    threshold = 128
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            threshold = i
    return threshold


def _palette_image(palette: list[tuple[int, int, int]]) -> Image.Image:
    p = Image.new("P", (1, 1))
    flat: list[int] = []
    for r, g, b in palette:
        flat.extend([r, g, b])
    last = palette[-1] if palette else (0, 0, 0)
    while len(flat) < 256 * 3:
        flat.extend(last)
    p.putpalette(flat[: 256 * 3])
    return p


def kasun_with_accent(*args, **kwargs):  # noqa: D401
    """Deprecated — use kasun_color() instead."""
    return kasun_color(*args, **kwargs)
