"""Render a quick PIL collage from a WeekSummary.

This image acts as the *seed* for img2img — Stable Diffusion will redraw it
in the bad-doodle style. Even a crude collage works because the prompt
asks for a recognizable-but-distorted result.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from src.diary import WeekSummary


WEATHER_GLYPH = {
    "sunny": "*",
    "cloudy": "~",
    "rainy": "/",
    "windy": ">",
    "snowy": "+",
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render(summary: WeekSummary, size: int = 512) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(20)
    body_font = _font(14)

    draw.text((16, 12), "weekly postcard", fill="black", font=title_font)
    draw.text(
        (16, 38),
        f"{summary.week_start} ~ {summary.week_end}",
        fill="black",
        font=body_font,
    )

    # Water cups row
    cups = max(1, min(summary.water_event_count, 8))
    cup_w = (size - 32) // 8
    base_y = 80
    for i in range(cups):
        x = 16 + i * cup_w
        draw.rectangle(
            [x + 4, base_y + 18, x + cup_w - 8, base_y + 60], outline="black", width=2
        )
        draw.line(
            [x + 4, base_y + 18, x + cup_w - 8, base_y + 18], fill="black", width=2
        )
        draw.ellipse(
            [x + 8, base_y + 14, x + cup_w - 12, base_y + 22], outline="black", width=1
        )

    # Weather glyph block
    glyph = WEATHER_GLYPH.get(summary.dominant_weather, "?")
    weather_y = base_y + 80
    draw.text((16, weather_y), f"weather: {summary.dominant_weather}", fill="black", font=body_font)
    for i in range(7):
        draw.text(
            (16 + i * 32, weather_y + 24),
            glyph,
            fill="black",
            font=_font(28),
        )

    # Places list
    places_y = weather_y + 70
    draw.text((16, places_y), "places:", fill="black", font=body_font)
    for i, name in enumerate(summary.top_places[:5]):
        draw.text(
            (16, places_y + 22 + i * 18),
            f"- {name}",
            fill="black",
            font=body_font,
        )

    # Avg temp footer
    draw.text(
        (16, size - 30),
        f"avg {summary.avg_temp_c} C, water {summary.total_water_ml} ml",
        fill="black",
        font=body_font,
    )

    return img
