"""Apply the kasun (한심함) filter at four intensity levels to existing
samples — no new SD inference, just pixel ops. Lets us see how close we
can push SD-Turbo output toward the actual 하찮은 프롬프트 GPT-4o look
without changing the model.

Sources cover three regimes:
  - already-bw doodle (random_w11_doodle, txt2img and img2img)
  - colorful painterly (cross_dreamlike_diffusion)
  - photographic (quality_realistic) — worst case for the filter

For each source, four kasun outputs:
  light    grid=64, bilevel
  medium   grid=48, bilevel
  heavy    grid=32, bilevel
  extreme  grid=24, bilevel + 1 px tremor jitter
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kasun_filter import kasun  # noqa: E402

SAMPLES = ROOT / "samples"


SOURCES = [
    "random_w11_doodle.png",
    "random_w11_txt2img.png",
    "cross_dreamlike_diffusion.png",
    "quality_realistic.png",
    "viral_friends_mom_portrait.png",
]

LEVELS = [
    {"name": "light", "grid": 64, "threshold": None, "jitter": 0},
    {"name": "medium", "grid": 48, "threshold": None, "jitter": 0},
    {"name": "heavy", "grid": 32, "threshold": None, "jitter": 0},
    {"name": "extreme", "grid": 24, "threshold": None, "jitter": 1},
]


def main() -> None:
    rows: list[dict] = []
    for src_name in SOURCES:
        src_path = SAMPLES / src_name
        if not src_path.exists():
            print(f"[kasun] missing source: {src_name} — skipping")
            continue

        stem = src_path.stem
        img = Image.open(src_path)
        out_paths: dict[str, str] = {}

        for lv in LEVELS:
            out = kasun(
                img,
                grid=lv["grid"],
                colors=2,
                threshold=lv["threshold"],
                output_size=1024,
                jitter=lv["jitter"],
                jitter_seed=42,
            )
            out_path = SAMPLES / f"kasun_{lv['name']}_{stem}.png"
            out.save(out_path)
            out_paths[lv["name"]] = out_path.name

        rows.append({"src": src_name, "outputs": out_paths})
        print(f"[kasun] {src_name} -> {', '.join(out_paths.values())}")

    # Index page
    headers = "| source | " + " | ".join(lv["name"] for lv in LEVELS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in LEVELS) + " |"
    lines = [
        "# Kasun (한심함) filter ladder",
        "",
        "Pure pixel-op post-process to push any image toward the 1-bit",
        "MS Paint feel of the original 하찮은 프롬프트 GPT-4o output.",
        "",
        "Levels:",
        "- **light** : 64×64 grid, threshold 200 → cleaner doodle",
        "- **medium**: 48×48 grid, threshold 180 → chunky bitmap",
        "- **heavy** : 32×32 grid, threshold 180 → MS Paint",
        "- **extreme**: 24×24 grid, threshold 180, +1 px row jitter (mouse tremor)",
        "",
        "All outputs are 1024×1024 with nearest-neighbor upscale so the",
        "blocks read clearly.",
        "",
        headers,
        sep,
    ]
    for r in rows:
        cell_src = f"![]({r['src']})"
        cells = [f"![]({r['outputs'][lv['name']]})" for lv in LEVELS]
        lines.append(f"| {cell_src} | " + " | ".join(cells) + " |")
    (SAMPLES / "kasun_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[kasun] index -> samples/kasun_grid.md")


if __name__ == "__main__":
    main()
