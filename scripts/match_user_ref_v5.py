"""v5 — minimal post-process, let SD-Turbo carry the doodle aesthetic.

Real takeaway from comparing user's ChatGPT result side-by-side:
  - The trend's actual look is HAND-DRAWN (wobbly, shaky lines, pastel
    fills) NOT 1-bit pixel-perfect. v4 went too far toward "8-bit RPG
    sprite" because rembg stripped the BG and kasun_color pixelised.
  - ChatGPT keeps the WHOLE SCENE — cafe table, plant, hanging lights,
    other people in faint pencil — as a single doodle. Stripping BG is
    the wrong move.
  - SD-Turbo's natural cartoon output was actually closer to the
    target than any of our heavy post-processed versions.

So v5 inverts the strategy: minimal post-process, lean on SD-Turbo
with prompt language that targets shaky mouse-drawn doodle directly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generator  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


# Prompt: target the "drawn-with-a-mouse-in-MS-Paint" aesthetic
# explicitly. Wobbly/shaky/uneven lines + crayon/marker fill, NOT
# pixel art. Keep the whole scene.
PROMPT = (
    "shaky hand-drawn doodle made with a computer mouse in MS Paint, "
    "wobbly uneven lines, crayon-coloured fill, child-like drawing, "
    "thin pen strokes, pastel colours, awkward proportions, "
    "naive amateur sketch, full scene with figure and surroundings"
)
NEGATIVE = (
    "photorealistic, sharp, smooth, professional, anti-aliased, gradient, "
    "vector art, pixel art, 8-bit, sprite, polished, hd, oil painting"
)


def soft_finish(img: Image.Image) -> Image.Image:
    """Mild 'paper' finish — reduce saturation slightly so colours read
    as crayon/marker rather than digital RGB. No edge filtering, no
    pixelisation, no BG removal."""
    out = ImageEnhance.Color(img).enhance(0.85)  # very gentle desaturate
    return out


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    if not src_path.exists():
        print(f"missing {src_path}")
        return

    photo = Image.open(src_path).convert("RGB")
    print(f"[v5] photo {photo.size}")

    # Three strength levels, see which matches the target best.
    results: list[tuple[str, Image.Image]] = []
    for tag, strength, steps in (
        ("s55", 0.55, 4),
        ("s70", 0.70, 4),
        ("s85", 0.85, 4),
    ):
        print(f"[v5] SD-Turbo img2img {tag} strength={strength}...")
        t = time.time()
        sd_out = generator.generate_img2img(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            init_image=photo,
            steps=steps,
            strength=strength,
            seed=42,
        )
        print(f"[v5]    SD done in {time.time() - t:.1f}s")
        sd_out.save(SAMPLES / f"v5_set1_sd_{tag}.png")

        finished = soft_finish(sd_out)
        finished.save(SAMPLES / f"v5_set1_final_{tag}.png")
        results.append((tag, finished))

    # Index page
    md = [
        "# v5 — minimal post-process, lean on SD-Turbo prompt",
        "",
        "v4 (img2img + rembg + kasun_color) went too far: looked like an",
        "8-bit sprite, not the trend's hand-drawn doodle. v5 strips the",
        "post-process down to a tiny desaturation pass and leans on the",
        "SD-Turbo prompt to hit the wobbly-mouse-drawn aesthetic.",
        "",
        "| stage | image |",
        "| --- | --- |",
        f"| input | ![](reference/user_test/line/set1_input.png) |",
    ]
    for tag, _ in results:
        md.append(f"| v5 SD strength={tag[1:]}% | ![](v5_set1_sd_{tag}.png) |")
        md.append(f"| v5 final ({tag}) — soft finish | ![](v5_set1_final_{tag}.png) |")
    md.append(f"| target — ChatGPT 하찮은 | ![](reference/user_test/line/set1_output.png) |")
    (SAMPLES / "match_v5_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v5] index -> samples/match_v5_grid.md")


if __name__ == "__main__":
    main()
