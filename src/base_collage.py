"""Render a quick PIL collage from a WeekSummary.

This image acts as the *seed* for img2img — Stable Diffusion will redraw it
in the bad-doodle style. The collage is intentionally pictographic and has
NO TEXT, because text on the seed gets reinterpreted by SD as garbled fake
text in the output.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from src.diary import WeekSummary


CATEGORY_SHAPE = {
    "cafe": "cup",
    "park": "tree",
    "work": "building",
    "bookstore": "book",
    "restaurant": "bowl",
    "landmark": "tower",
    "market": "cart",
    "home": "house",
}


def _draw_cup(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.rectangle([x + 2, y + h * 0.25, x + w - 4, y + h], outline="black", width=2)
    d.ellipse([x + 4, y + h * 0.2, x + w - 6, y + h * 0.35], outline="black", width=2)
    # handle
    d.arc([x + w - 8, y + h * 0.4, x + w + 6, y + h * 0.85], -90, 90, fill="black", width=2)


def _draw_sun(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=3)
    for i in range(8):
        a = i * math.pi / 4
        x1 = cx + int(math.cos(a) * (r + 4))
        y1 = cy + int(math.sin(a) * (r + 4))
        x2 = cx + int(math.cos(a) * (r + 14))
        y2 = cy + int(math.sin(a) * (r + 14))
        d.line([x1, y1, x2, y2], fill="black", width=2)


def _draw_cloud(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    d.ellipse([cx - r, cy - r // 2, cx, cy + r // 2], outline="black", width=2)
    d.ellipse([cx - r // 2, cy - r, cx + r // 2, cy], outline="black", width=2)
    d.ellipse([cx, cy - r // 2, cx + r, cy + r // 2], outline="black", width=2)


def _draw_rain(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    _draw_cloud(d, cx, cy, r)
    for i in range(5):
        x = cx - r + i * (r // 2)
        d.line([x, cy + r // 2 + 4, x - 4, cy + r // 2 + 16], fill="black", width=2)


def _draw_tree(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    d.line([cx, y + h, cx, y + h // 2], fill="black", width=2)
    d.polygon(
        [(cx - w // 2, y + h // 2), (cx + w // 2, y + h // 2), (cx, y)],
        outline="black",
    )


def _draw_house(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.rectangle([x, y + h // 3, x + w, y + h], outline="black", width=2)
    d.polygon(
        [(x, y + h // 3), (x + w, y + h // 3), (x + w // 2, y)], outline="black"
    )


def _draw_building(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.rectangle([x, y, x + w, y + h], outline="black", width=2)
    rows, cols = 3, 3
    for r in range(rows):
        for c in range(cols):
            wx = x + 4 + c * (w // cols)
            wy = y + 4 + r * (h // rows)
            d.rectangle([wx, wy, wx + 8, wy + 8], outline="black", width=1)


def _draw_book(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    for i in range(3):
        d.rectangle(
            [x, y + i * (h // 3), x + w, y + (i + 1) * (h // 3) - 2],
            outline="black",
            width=2,
        )


def _draw_bowl(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.arc([x, y + h // 4, x + w, y + h], 0, 180, fill="black", width=2)
    d.line([x, y + h // 2, x + w, y + h // 2], fill="black", width=2)


def _draw_tower(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    cx = x + w // 2
    d.line([cx - w // 4, y + h, cx + w // 4, y + h], fill="black", width=2)
    d.line([cx - w // 6, y + h, cx - w // 12, y + h // 3], fill="black", width=2)
    d.line([cx + w // 6, y + h, cx + w // 12, y + h // 3], fill="black", width=2)
    d.line([cx - w // 12, y + h // 3, cx + w // 12, y + h // 3], fill="black", width=2)
    d.polygon(
        [(cx - w // 4, y + h // 3), (cx + w // 4, y + h // 3), (cx, y)],
        outline="black",
    )


def _draw_cart(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.rectangle([x + 4, y + h // 3, x + w - 4, y + 2 * h // 3], outline="black", width=2)
    d.line([x, y + h // 3, x + 4, y + h // 3], fill="black", width=2)
    d.ellipse([x + 4, y + 2 * h // 3, x + 14, y + 2 * h // 3 + 10], outline="black", width=2)
    d.ellipse([x + w - 14, y + 2 * h // 3, x + w - 4, y + 2 * h // 3 + 10], outline="black", width=2)


SHAPE_DRAWERS = {
    "cup": _draw_cup,
    "tree": _draw_tree,
    "building": _draw_building,
    "book": _draw_book,
    "bowl": _draw_bowl,
    "tower": _draw_tower,
    "cart": _draw_cart,
    "house": _draw_house,
}


def _draw_weather(
    d: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, r: int
) -> None:
    if kind == "sunny":
        _draw_sun(d, cx, cy, r)
    elif kind == "cloudy":
        _draw_cloud(d, cx, cy, r)
    elif kind == "rainy":
        _draw_rain(d, cx, cy, r)
    elif kind == "windy":
        for i in range(3):
            d.line(
                [cx - r, cy - r + i * (r // 2), cx + r, cy - r + i * (r // 2)],
                fill="black",
                width=2,
            )
    elif kind == "snowy":
        for i in range(5):
            x = cx - r + i * (r // 2)
            d.text((x, cy), "*", fill="black")
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=2)


def render(summary: WeekSummary, size: int = 512) -> Image.Image:
    """Render a TEXT-FREE pictographic collage as the img2img seed."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Top-right: weather icon (large)
    _draw_weather(draw, summary.dominant_weather, cx=size - 90, cy=90, r=46)

    # Water cups: cap at 16 so they fit in two rows of 8 inside 512 width.
    # The prompt builder caps at 24, so the count visible here is at most 16
    # while the prompt may say up to 24 — fine, both are upper bounds and
    # the SD model approximates either way.
    cups = max(1, min(summary.water_event_count, 16))
    cup_w = 48
    cup_h = 56
    cups_per_row = 8
    rows_needed = (cups + cups_per_row - 1) // cups_per_row
    row_y0 = 190
    for idx in range(cups):
        r = idx // cups_per_row
        c = idx % cups_per_row
        cups_in_this_row = min(cups - r * cups_per_row, cups_per_row)
        total_w = cups_in_this_row * cup_w + (cups_in_this_row - 1) * 12
        start_x = (size - total_w) // 2
        x = start_x + c * (cup_w + 12)
        y = row_y0 + r * (cup_h + 14)
        _draw_cup(draw, x, y, cup_w, cup_h)

    # When two rows of cups push the place icons down, also push the grid
    grid_y_offset = (rows_needed - 1) * (cup_h + 14)

    # Bottom area: 2x4 grid of place icons (shifted down if cup rows used 2)
    cats = list(summary.place_categories.keys())[:8]
    grid_y = 320 + grid_y_offset
    cell_w, cell_h = 100, 80
    for idx, cat in enumerate(cats):
        col = idx % 4
        row = idx // 4
        x = 30 + col * (cell_w + 10)
        y = grid_y + row * (cell_h + 12)
        shape = CATEGORY_SHAPE.get(cat)
        drawer = SHAPE_DRAWERS.get(shape) if shape else None
        if drawer:
            drawer(draw, x, y, cell_w - 16, cell_h)

    return img
