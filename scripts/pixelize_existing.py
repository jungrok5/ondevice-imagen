"""Apply pixelize to existing colorful samples — no new SD inference.

Two modes per source image:
  1. 16-color median-cut    (preserves source palette, vibrant retro)
  2. Game Boy 4-color       (forces 1989 DMG-01 green palette)

Outputs land in samples/ with names like:
  pixel16_<source_stem>.png
  pixelgb_<source_stem>.png

Plus an index page at samples/pixelize_grid.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pixelize import pixelize, pixelize_with_palette, GAMEBOY_PALETTE  # noqa: E402

SAMPLES = ROOT / "samples"


SOURCES = [
    # The new viral winner (round 2)
    "drill_mom_korean_living_room.png",
    # Quality dial endpoints
    "quality_realistic.png",
    "quality_anime.png",
    # Painterly aesthetics
    "style_watercolor_diary.png",
    "cross_dreamlike_diffusion.png",
    # Round-1 viral candidates
    "viral_hotel_art_70s.png",
    "viral_friends_mom_portrait.png",
]


def main() -> None:
    rows: list[dict] = []
    for src_name in SOURCES:
        src_path = SAMPLES / src_name
        if not src_path.exists():
            print(f"[pixelize] missing source: {src_name} — skipping")
            continue

        stem = src_path.stem
        img = Image.open(src_path)

        out16 = pixelize(img, grid=64, colors=16)
        out16_path = SAMPLES / f"pixel16_{stem}.png"
        out16.save(out16_path)

        out_gb = pixelize_with_palette(img, grid=64, palette=GAMEBOY_PALETTE)
        out_gb_path = SAMPLES / f"pixelgb_{stem}.png"
        out_gb.save(out_gb_path)

        rows.append({"src": src_name, "p16": out16_path.name, "gb": out_gb_path.name})
        print(f"[pixelize] {src_name} -> {out16_path.name}, {out_gb_path.name}")

    # Index page
    lines = [
        "# Pixelize gallery",
        "",
        "Existing colorful samples remapped through `src/pixelize.py`. No SD",
        "inference — pure PIL post-process.",
        "",
        "- `pixel16_*.png`: 64×64 grid, 16-color median-cut palette picked per",
        "  image. Preserves source colors with a retro paint-program feel.",
        "- `pixelgb_*.png`: 64×64 grid, fixed 4-color Game Boy DMG-01 palette.",
        "  Forces a 1989 game-screen aesthetic regardless of source colors.",
        "",
        "| source | 16-color | Game Boy |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| ![]({r['src']}) | ![]({r['p16']}) | ![]({r['gb']}) |"
        )
    (SAMPLES / "pixelize_grid.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\n[pixelize] index -> samples/pixelize_grid.md")


if __name__ == "__main__":
    main()
