"""True pixel-art post-processing.

SD output looks 'pixel-ish' but has anti-aliased edges, gradients, and a
full RGB palette. Real pixel art has hard edges and a fixed palette.
This module enforces that: downsample to a small grid, quantize colors,
upsample with nearest-neighbor.

Two quantization modes:
  - pixelize()              : median-cut to N colors picked per-image.
                              Preserves the source palette.
  - pixelize_with_palette() : remap to a fixed palette (e.g. Game Boy
                              4-color, NES). Forces a retro game look
                              regardless of source colors.
"""
from __future__ import annotations

from PIL import Image


# Classic 1989 Game Boy 4-color green palette (DMG-01).
GAMEBOY_PALETTE = [
    (15, 56, 15),     # darkest
    (48, 98, 48),     # dark
    (139, 172, 15),   # light
    (155, 188, 15),   # lightest
]

# A reduced NES-ish 16-color palette suitable for fast quantize.
NES_PALETTE = [
    (0, 0, 0),
    (255, 255, 255),
    (124, 124, 124),
    (188, 188, 188),
    (172, 16, 0),
    (236, 92, 92),
    (228, 92, 16),
    (252, 160, 68),
    (40, 96, 16),
    (40, 188, 76),
    (60, 64, 168),
    (108, 168, 252),
    (160, 0, 84),
    (236, 64, 168),
    (252, 252, 0),
    (140, 80, 32),
]


def pixelize(
    img: Image.Image,
    grid: int = 128,
    colors: int = 16,
    output_size: int = 1024,
) -> Image.Image:
    """Force a pixel-perfect look on an SD output.

    grid:        target small-image edge in pixels. 128 ≈ a 16-bit RPG
                 native resolution (e.g. Chrono Trigger ~256x224). Smaller
                 grid = chunkier pixels.
    colors:      palette size after median-cut quantization. 16 reads as
                 "limited 16-color paint program".
    output_size: edge length of the final upscaled PNG in pixels. 1024
                 makes each block clearly visible on a HiDPI display.
    """
    src = img.convert("RGB")

    small = src.resize((grid, grid), Image.LANCZOS)
    quantized = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantized = quantized.convert("RGB")

    return quantized.resize((output_size, output_size), Image.NEAREST)


def _palette_image(palette: list[tuple[int, int, int]]) -> Image.Image:
    """Build a mode-P image whose palette table holds the given colors."""
    p = Image.new("P", (1, 1))
    flat: list[int] = []
    for r, g, b in palette:
        flat.extend([r, g, b])
    # Pad up to 256 entries by repeating the last color so unused slots
    # don't get matched to (0,0,0) by Pillow's quantizer.
    last = palette[-1] if palette else (0, 0, 0)
    while len(flat) < 256 * 3:
        flat.extend(last)
    p.putpalette(flat[: 256 * 3])
    return p


def pixelize_with_palette(
    img: Image.Image,
    grid: int = 128,
    palette: list[tuple[int, int, int]] | None = None,
    output_size: int = 1024,
) -> Image.Image:
    """Pixelize and remap to a fixed palette.

    If palette is None, falls back to pixelize() with 16 median-cut colors.
    """
    if palette is None:
        return pixelize(img, grid=grid, colors=16, output_size=output_size)

    src = img.convert("RGB")

    small = src.resize((grid, grid), Image.LANCZOS)
    quantized = small.quantize(
        palette=_palette_image(palette),
        dither=Image.Dither.NONE,
    ).convert("RGB")

    return quantized.resize((output_size, output_size), Image.NEAREST)
