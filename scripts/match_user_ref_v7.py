"""v7 — combined filter targeting the user's preferred aesthetic.

User picked v5 strength=0.85 as their preferred style: colourful
cartoon with thick black outlines, full cafe scene preserved. v6
stylize_60_045 hit the same register but cleaner (subject identity
preserved). v7 = stylize_60_045 + per-row pixel jitter to add the
mouse-tremor wobble we still don't have.

Pipeline:
  photo
   -> SD-Turbo img2img strength=0.75 with cartoon prompt
   -> cv2.stylization (sigma_s=60, sigma_r=0.45) — thick edges, flat fill
   -> random per-row horizontal jitter ±jit_px — fakes mouse shake
   -> light saturation tweak
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generator  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


PROMPT = (
    "simple flat cartoon illustration, naive style, thin black outlines, "
    "pastel colour fill, awkward proportions, child drawing"
)
NEGATIVE = (
    "photorealistic, detailed, sharp focus, oil painting, smooth gradient, "
    "professional, hd, vector art, pixel art"
)


def stylize(pil_img: Image.Image, sigma_s: float, sigma_r: float) -> Image.Image:
    arr = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    out = cv2.stylization(bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def apply_jitter(img: Image.Image, max_px: int, seed: int = 0) -> Image.Image:
    """Shift each row horizontally by a random ±max_px to fake the wobble
    of a hand drawing with a mouse. Fills exposed margins with white."""
    rng = random.Random(seed)
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    out = np.full_like(arr, 255)  # start with white canvas
    for y in range(h):
        dx = rng.randint(-max_px, max_px)
        if dx == 0:
            out[y] = arr[y]
        elif dx > 0:
            out[y, dx:] = arr[y, : w - dx]
        else:  # dx < 0
            out[y, : w + dx] = arr[y, -dx:]
    return Image.fromarray(out)


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    photo = Image.open(src_path).convert("RGB")
    print(f"[v7] photo {photo.size}")

    print(f"[v7] SD-Turbo img2img strength=0.75...")
    t = time.time()
    sd = generator.generate_img2img(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        init_image=photo,
        steps=4,
        strength=0.75,
        seed=42,
    )
    print(f"[v7]    SD done in {time.time() - t:.1f}s")
    sd.save(SAMPLES / "v7_set1_sd.png")

    # Apply cv2 stylization
    stylized = stylize(sd, sigma_s=60, sigma_r=0.45)
    stylized.save(SAMPLES / "v7_set1_stylized.png")

    # Apply jitter at three intensities to find the right wobble
    for tag, jit in (("j1", 1), ("j2", 2), ("j3", 3)):
        wobbly = apply_jitter(stylized, max_px=jit, seed=7)
        # Light saturation softening so colours read crayon-like
        wobbly = ImageEnhance.Color(wobbly).enhance(0.9)
        wobbly.save(SAMPLES / f"v7_set1_final_{tag}.png")

    md = [
        "# v7 — cv2 stylization + row-jitter wobble",
        "",
        "User-preferred direction (per v5_s85 feedback): colourful cartoon",
        "with thick black outlines, full scene. v7 takes v6 stylize_60_045",
        "and adds mouse-tremor wobble via per-row horizontal jitter.",
        "",
        "| stage | image |",
        "| --- | --- |",
        f"| input | ![](reference/user_test/line/set1_input.png) |",
        f"| SD-Turbo img2img (s=0.75) | ![](v7_set1_sd.png) |",
        f"| + cv2.stylization | ![](v7_set1_stylized.png) |",
        f"| + jitter ±1 px | ![](v7_set1_final_j1.png) |",
        f"| + jitter ±2 px | ![](v7_set1_final_j2.png) |",
        f"| + jitter ±3 px | ![](v7_set1_final_j3.png) |",
        f"| **target — ChatGPT 하찮은** | ![](reference/user_test/line/set1_output.png) |",
        f"| user-favourite — v5 s=0.85 | ![](v5_set1_sd_s85.png) |",
    ]
    (SAMPLES / "match_v7_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v7] index -> samples/match_v7_grid.md")


if __name__ == "__main__":
    main()
