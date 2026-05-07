"""v6 — cv2 stylization filters on SD-Turbo output.

v5 showed SD-Turbo can't natively produce the "shaky mouse drawing"
look — it leans painterly. v4's kasun_color went too far the other
way (8-bit sprite). The middle ground is to keep SD's general
composition + colour layout but apply OpenCV's purpose-built
artistic filters that target hand-drawn / sketched aesthetics:

  - cv2.stylization  — edge-preserving smoothing, cartoon flatness
  - cv2.pencilSketch — actual pencil-line + light colour wash

cv2 was installed transitively by rembg, so we already have the
`opencv-python-headless` package.
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

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


# Same prompt shape as v4 but stripped of "MS Paint" cues that pushed
# v5 toward painterly. Want a colourful cartoon out of SD; cv2 will
# add the hand-drawn finish.
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


def pencil_color(
    pil_img: Image.Image,
    sigma_s: float,
    sigma_r: float,
    shade: float,
) -> Image.Image:
    arr = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _gray, color = cv2.pencilSketch(
        bgr, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade
    )
    return Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    photo = Image.open(src_path).convert("RGB")
    print(f"[v6] photo {photo.size}")

    # SD-Turbo img2img — moderate strength, simple cartoon prompt.
    print(f"[v6] SD-Turbo img2img...")
    t = time.time()
    sd = generator.generate_img2img(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        init_image=photo,
        steps=4,
        strength=0.70,
        seed=42,
    )
    print(f"[v6]    SD done in {time.time() - t:.1f}s")
    sd_path = SAMPLES / "v6_set1_sd.png"
    sd.save(sd_path)

    # Apply cv2 filters at various parameter sweeps.
    variants: list[tuple[str, Image.Image]] = []

    # Stylization sweep
    for tag, ss, sr in [
        ("stylize_60_045", 60, 0.45),
        ("stylize_100_06", 100, 0.6),
        ("stylize_150_07", 150, 0.7),
    ]:
        out = stylize(sd, ss, sr)
        out.save(SAMPLES / f"v6_set1_{tag}.png")
        variants.append((tag, out))

    # Pencil sketch (color)
    for tag, ss, sr, sh in [
        ("pencil_60_07_005", 60, 0.07, 0.05),
        ("pencil_100_01_010", 100, 0.1, 0.1),
        ("pencil_30_05_005", 30, 0.5, 0.05),
    ]:
        out = pencil_color(sd, ss, sr, sh)
        out.save(SAMPLES / f"v6_set1_{tag}.png")
        variants.append((tag, out))

    # Index page
    md = [
        "# v6 — cv2 stylization / pencilSketch on SD-Turbo output",
        "",
        "v5 showed SD-Turbo can't make the shaky-mouse-drawn look on its",
        "own. v6 keeps SD-Turbo's natural cartoon output and applies",
        "OpenCV's purpose-built hand-drawn filters.",
        "",
        "| variant | image |",
        "| --- | --- |",
        f"| input photo | ![](reference/user_test/line/set1_input.png) |",
        f"| SD-Turbo (cartoon, no filter) | ![](v6_set1_sd.png) |",
    ]
    for tag, _ in variants:
        md.append(f"| {tag} | ![](v6_set1_{tag}.png) |")
    md.append(f"| **target — ChatGPT 하찮은** | ![](reference/user_test/line/set1_output.png) |")
    (SAMPLES / "match_v6_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[v6] index -> samples/match_v6_grid.md")


if __name__ == "__main__":
    main()
