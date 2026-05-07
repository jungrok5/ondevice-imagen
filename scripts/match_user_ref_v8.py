"""v8 — prompt-variation sweep on the v7 pipeline.

Same SD-Turbo img2img (s=0.75) + cv2.stylization + ±3 px row jitter
as v7. Only the *prompt wording* varies. Five candidates targeting
different angles of "child mouse-drawn doodle":

  p1 — direct child + mouse
  p2 — wrong-hand five-second sketch
  p3 — MS Paint 1995 amateur
  p4 — kindergarten crayon
  p5 — intentionally-bad meta instruction
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generator  # noqa: E402
from scripts.match_user_ref_v7 import stylize, apply_jitter  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


PROMPTS = {
    "p1_child_mouse": (
        "drawn by a five-year-old child with a computer mouse, awkward "
        "shaky lines, wobbly amateur scribble, white paper, simple flat "
        "colour fill"
    ),
    "p2_wrong_hand": (
        "sketched in five seconds with the wrong hand, terrible mouse "
        "drawing, uneven proportions, naive flat fill, white paper"
    ),
    "p3_ms_paint_1995": (
        "Microsoft Paint amateur drawing from 1995, crude pixel-mouse "
        "scribble, awkward shapes, low effort, flat saturated colours"
    ),
    "p4_kindergarten_crayon": (
        "kindergarten crayon drawing on white paper, naive figure, "
        "simple shapes, child art, thick wax-crayon lines, pastel fill"
    ),
    "p5_intentionally_bad": (
        "intentionally awful sketch, deliberately badly drawn, ugly "
        "proportions, simple cartoon with thick black outlines, "
        "white background"
    ),
}

NEGATIVE = (
    "photorealistic, detailed, sharp focus, oil painting, smooth gradient, "
    "professional, hd, vector art, pixel art, polished, masterpiece"
)

STRENGTH = 0.75
SEED = 42


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    photo = Image.open(src_path).convert("RGB")
    print(f"[v8] photo {photo.size}")

    md = [
        "# v8 — prompt variation sweep on the v7 pipeline",
        "",
        "Same pipeline as v7 (img2img s=0.75 -> cv2.stylization -> "
        "row-jitter ±3 px). Only the prompt wording differs.",
        "",
        "| variant | prompt | image |",
        "| --- | --- | --- |",
        f"| input | — | ![](reference/user_test/line/set1_input.png) |",
    ]

    for tag, prompt in PROMPTS.items():
        print(f"\n[v8] {tag}: {prompt[:80]}...")
        t = time.time()
        sd = generator.generate_img2img(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            init_image=photo,
            steps=4,
            strength=STRENGTH,
            seed=SEED,
        )
        print(f"[v8]    SD done in {time.time() - t:.1f}s")

        stylized = stylize(sd, sigma_s=60, sigma_r=0.45)
        wobbly = apply_jitter(stylized, max_px=3, seed=7)
        wobbly = ImageEnhance.Color(wobbly).enhance(0.9)

        sd_path = SAMPLES / f"v8_{tag}_sd.png"
        sd.save(sd_path)
        final_path = SAMPLES / f"v8_{tag}_final.png"
        wobbly.save(final_path)

        # Truncate prompt for table
        short_prompt = prompt[:60] + ("..." if len(prompt) > 60 else "")
        md.append(f"| {tag} | `{short_prompt}` | ![]({final_path.name}) |")

    md.append(f"| **target — ChatGPT 하찮은** | — | ![](reference/user_test/line/set1_output.png) |")
    md.append(f"| user-favourite — v5 s=0.85 | — | ![](v5_set1_sd_s85.png) |")

    (SAMPLES / "match_v8_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[v8] index -> samples/match_v8_grid.md")


if __name__ == "__main__":
    main()
