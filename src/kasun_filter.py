"""한심함 (kasun) maximizer — push SD output toward the 하찮은 프롬프트 look.

The original GPT-4o output of that trend looks the way it does because
every line is hard-edged 1-bit bitmap, the palette collapses to black
ink on white paper, and shapes are visibly recognisable but distorted.
SD-Turbo gets the *shapes* right but ships anti-aliased lines and a
full RGB palette; we have to force the rest with post-processing.

Three operations stacked:
  1. downsample to a chunky grid (forces pixel visibility)
  2. bilevel threshold or tiny palette (forces hard edges, kills gradient)
  3. nearest-neighbor upscale (no smoothing — every pixel a hard block)

All pure pixel ops → trivial to port to iOS Core Image / Android Bitmap
APIs alongside the SD-Turbo Core ML / ONNX pipeline. No extra ML model.
"""
from __future__ import annotations

import random

from PIL import Image, ImageOps


def kasun(
    img: Image.Image,
    grid: int = 48,
    colors: int = 2,
    threshold: int | None = None,
    output_size: int = 1024,
    jitter: int = 0,
    jitter_seed: int | None = None,
    auto_invert: bool = True,
) -> Image.Image:
    """Aggressive deliberate-bad post-process.

    grid:        small-image edge before quantize+upscale. 48 ≈ chunky
                 MS Paint feel; 24 ≈ Game Boy 2-bit look.
    colors:      palette size. 2 = pure bilevel B/W (most 하찮은).
                 3-4 keeps a tiny accent palette.
    threshold:   used only when colors == 2. 0..255 cutoff for B/W.
                 None → auto-pick from per-image mean luminance.
    output_size: final upscaled size in pixels. Nearest-neighbor.
    jitter:      pixels of horizontal hand-tremor displacement applied
                 BEFORE quantize. 0 disables. 1-3 looks like mouse-shake.
    jitter_seed: optional seed for the jitter rng so the shake is
                 reproducible per call.
    auto_invert: when colors == 2, flip the thresholded image if the
                 source is dark-dominated, so the output is always
                 black-on-white instead of white-on-black.
    """
    src = img.convert("RGB")

    if jitter > 0:
        src = _apply_jitter(src, jitter, jitter_seed)

    small = src.resize((grid, grid), Image.LANCZOS)

    if colors <= 2:
        # First normalize luminance so threshold lands sensibly regardless
        # of whether the source is light or dark dominated.
        gray = ImageOps.autocontrast(small.convert("L"))
        if auto_invert:
            mean = sum(gray.getdata()) / (gray.width * gray.height)
            if mean < 110:  # dark dominated → flip so white is the background
                gray = ImageOps.invert(gray)

        thr = threshold if threshold is not None else _otsu_threshold(gray)
        bilevel = gray.point(lambda p: 255 if p > thr else 0, mode="1")
        out_small = bilevel.convert("RGB")
    else:
        out_small = small.quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT
        ).convert("RGB")

    return out_small.resize((output_size, output_size), Image.NEAREST)


def _otsu_threshold(gray: Image.Image) -> int:
    """Pick a threshold that maximizes between-class variance (Otsu's method)."""
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


def _apply_jitter(img: Image.Image, magnitude: int, seed: int | None) -> Image.Image:
    """Random per-row horizontal displacement to fake mouse-tremor lines."""
    rng = random.Random(seed)
    px = img.load()
    w, h = img.size
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out_px = out.load()
    for y in range(h):
        dx = rng.randint(-magnitude, magnitude)
        for x in range(w):
            sx = x - dx
            if 0 <= sx < w:
                out_px[x, y] = px[sx, y]
            else:
                out_px[x, y] = (255, 255, 255)
    return out


def kasun_with_accent(
    img: Image.Image,
    grid: int = 48,
    output_size: int = 1024,
    threshold: int = 180,
) -> Image.Image:
    """Bilevel B/W + a single accent color picked from the source.

    Closer to "MS Paint with one red highlight" than pure bilevel.
    """
    src = img.convert("RGB")
    small = src.resize((grid, grid), Image.LANCZOS)

    # Find the dominant non-grayscale color in the small image
    accent = _dominant_chroma(small)

    # Bilevel split for the lightness, then paint accent over the dark areas
    gray = small.convert("L")
    bilevel = gray.point(lambda p: 255 if p > threshold else 0).convert("RGB")

    # Replace black with accent IF accent is colored enough; otherwise stay B/W
    px = bilevel.load()
    w, h = bilevel.size
    if accent and _chroma(accent) > 60:
        for y in range(h):
            for x in range(w):
                if px[x, y] == (0, 0, 0):
                    px[x, y] = accent
    return bilevel.resize((output_size, output_size), Image.NEAREST)


def _dominant_chroma(img: Image.Image) -> tuple[int, int, int] | None:
    """Most common pixel that is *not* near-grayscale."""
    counts: dict[tuple[int, int, int], int] = {}
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if _chroma((r, g, b)) >= 40:  # has color
                key = (r // 32 * 32, g // 32 * 32, b // 32 * 32)
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _chroma(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return max(r, g, b) - min(r, g, b)
