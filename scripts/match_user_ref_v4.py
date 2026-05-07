"""4-stage pipeline matching the ChatGPT 하찮은 프롬프트 result.

  photo
    -> SD-Turbo img2img with the doodle prompt (preserves layout + identity)
    -> rembg (extract subject, clean white BG)
    -> kasun_color (flat colour fill + thick black outlines)
    -> final

This adds the 'abstract the scene' middle step our v3 was missing —
rembg gives us a real foreground/background mask without a heavyweight
segmentation network. ~25 MB U2-Net model, mobile-portable.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generator  # noqa: E402
from src.kasun_filter import kasun_color, kasun_line  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


# Stronger photo-fidelity prompt — keeps composition signal but biases
# colour/line work toward the doodle aesthetic. img2img with strength
# 0.82 leaves enough photo through that the subject identity survives.
PROMPT = (
    "naive child crayon drawing on white paper, thick black outlines, "
    "flat color fill, awkward childlike proportions, MS Paint doodle, "
    "deliberately badly drawn, pixelated low-res"
)
NEGATIVE = (
    "photorealistic, sharp, smooth, professional, high quality, "
    "anti-aliased, gradient, detailed scenery"
)
STRENGTH = 0.65  # lower → photo composition (sitting pose, coffee cup) survives
STEPS = 4
SEED = 42


def _on_white(rgba: Image.Image) -> Image.Image:
    """Composite an RGBA result onto a pure-white background."""
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba.convert("RGBA"))
    return bg.convert("RGB")


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    if not src_path.exists():
        print(f"missing {src_path}")
        return

    photo = Image.open(src_path).convert("RGB")
    print(f"[v4] photo loaded: {photo.size}")

    # 1. SD-Turbo img2img — keep enough of the photo signal that the
    #    subject is recognisable, but redraw in a doodle style.
    print(f"[v4] (1/3) SD-Turbo img2img strength={STRENGTH}...")
    t = time.time()
    sd_doodle = generator.generate_img2img(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        init_image=photo,
        steps=STEPS,
        strength=STRENGTH,
        seed=SEED,
    )
    print(f"[v4]    SD done in {time.time() - t:.1f}s")
    sd_path = SAMPLES / "v4_set1_sd.png"
    sd_doodle.save(sd_path)

    # 2. rembg — extract subject, transparent BG, then composite on white.
    print(f"[v4] (2/3) rembg subject extraction...")
    t = time.time()
    from rembg import remove  # imported lazily so the module-level import
    cut = remove(sd_doodle)
    cut_white = _on_white(cut)
    print(f"[v4]    rembg done in {time.time() - t:.1f}s")
    cut_path = SAMPLES / "v4_set1_cut.png"
    cut_white.save(cut_path)

    # 3. kasun_color — doodle aesthetic finish.
    print(f"[v4] (3/3) kasun_color finish...")
    kasun_out = kasun_color(
        cut_white,
        grid=128,
        colors=8,
        smooth=5,
        saturation=1.5,
        line_threshold=60,
        dilate=0,
        bg_to_white=False,  # rembg already gave us a clean white BG
    )
    kasun_path = SAMPLES / "v4_set1_kasun_color.png"
    kasun_out.save(kasun_path)

    # Side index page
    md = [
        "# kasun v4 (img2img + rembg + kasun_color) vs ChatGPT reference",
        "",
        "| stage | image |",
        "| --- | --- |",
        f"| 0. input photo | ![](reference/user_test/line/set1_input.png) |",
        f"| 1. SD-Turbo img2img (strength={STRENGTH}) | ![](v4_set1_sd.png) |",
        f"| 2. + rembg subject extraction | ![](v4_set1_cut.png) |",
        f"| 3. + kasun_color (final) | ![](v4_set1_kasun_color.png) |",
        f"| **target — ChatGPT 하찮은** | ![](reference/user_test/line/set1_output.png) |",
    ]
    (SAMPLES / "match_v4_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[v4] index -> samples/match_v4_grid.md")


if __name__ == "__main__":
    main()
