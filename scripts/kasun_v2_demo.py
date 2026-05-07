"""Apply the v2 kasun filters (line + color) to existing samples and
to the actual reference doodles, then save a side-by-side gallery.

Two filters:
  - kasun_line  : black single-line doodle on white (OpenAI-logo style)
  - kasun_color : 4-6 flat colours + thick black outlines (Sam Altman style)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kasun_filter import kasun_line, kasun_color  # noqa: E402

SAMPLES = ROOT / "samples"
REFERENCE = SAMPLES / "reference"


SOURCES = [
    # SD outputs
    "random_w11_doodle.png",
    "random_w11_txt2img.png",
    "random_w22_v1.png",
    "viral_friends_mom_portrait.png",
    "quality_realistic.png",
    "drill_mom_korean_living_room.png",
]


def main() -> None:
    rows = []
    for src_name in SOURCES:
        src_path = SAMPLES / src_name
        if not src_path.exists():
            print(f"[v2] missing: {src_name}")
            continue
        stem = src_path.stem
        img = Image.open(src_path)

        line_img = kasun_line(img)  # use defaults
        line_path = SAMPLES / f"v2_line_{stem}.png"
        line_img.save(line_path)

        color_img = kasun_color(img)  # use defaults
        color_path = SAMPLES / f"v2_color_{stem}.png"
        color_img.save(color_path)

        rows.append({"src": src_name, "line": line_path.name, "color": color_path.name})
        print(f"[v2] {src_name} -> {line_path.name}, {color_path.name}")

    # Reference doodles also passed through both filters for sanity check —
    # they should be near-identity (the references already are doodles).
    if (REFERENCE / "ref_heraldcorp_main.png").exists():
        for ref_name in ["ref_heraldcorp_main.png"]:
            stem = Path(ref_name).stem
            img = Image.open(REFERENCE / ref_name)
            line_path = SAMPLES / f"v2_line_{stem}.png"
            color_path = SAMPLES / f"v2_color_{stem}.png"
            kasun_line(img).save(line_path)
            kasun_color(img).save(color_path)
            rows.append({"src": f"reference/{ref_name}", "line": line_path.name, "color": color_path.name})
            print(f"[v2] {ref_name} -> {line_path.name}, {color_path.name}")

    # Index page
    lines = [
        "# kasun v2: 단색선 vs 컬러 낙서풍",
        "",
        "Two distinct sub-styles inside the 'doodle' look. Both are pure",
        "pixel-op post-process, no extra ML model.",
        "",
        "| source | 단색선 (line-only) | 컬러 낙서풍 (flat fill + outline) |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| ![]({r['src']}) | ![]({r['line']}) | ![]({r['color']}) |")
    (SAMPLES / "kasun_v2_grid.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\n[v2] index -> samples/kasun_v2_grid.md")


if __name__ == "__main__":
    main()
