"""True pixel-art post-processing.

SD output looks 'pixel-ish' but has anti-aliased edges, gradients, and a
full RGB palette. Real pixel art has hard edges and a fixed palette.
This module enforces that: downsample to a small grid, quantize colors,
upsample with nearest-neighbor.
"""
from __future__ import annotations

from PIL import Image


def pixelize(
    img: Image.Image,
    grid: int = 64,
    colors: int = 16,
) -> Image.Image:
    """Force a pixel-perfect look on an SD output.

    grid:    target small-image edge in pixels. 64 → each block is 8x8 in
             the upscaled output. Smaller grid = chunkier pixels.
    colors:  palette size after median-cut quantization. 16 reads as
             "limited 16-color paint program".
    """
    src = img.convert("RGB")
    w, h = src.size

    small = src.resize((grid, grid), Image.LANCZOS)
    quantized = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantized = quantized.convert("RGB")

    return quantized.resize((w, h), Image.NEAREST)
