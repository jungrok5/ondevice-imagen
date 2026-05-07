"""End-to-end pipeline matching the ChatGPT 하찮은 프롬프트 result.

Real photo input -> SD-Turbo img2img (doodle prompt) -> kasun_color.
This is the closest match to the ChatGPT pipeline (which does
abstraction + redraw in one step inside its own model).

Output:
  samples/match_set1_input.png      original photo
  samples/match_set1_sd.png         after SD-Turbo img2img
  samples/match_set1_kasun.png      after kasun_color
  samples/match_set1_ref.png        user's ChatGPT reference (for compare)
  samples/match_grid.md             side-by-side index
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import generator  # noqa: E402
from src.kasun_filter import kasun_line, kasun_color  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"


# Prompt mirrors the user's actual 하찮은 프롬프트 wording, in SD-friendly
# English with style tags front-loaded so they survive CLIP truncation.
PROMPT = (
    "naive child crayon drawing on plain white paper, isolated figures, "
    "thick black outlines, flat color fill, no background scenery, "
    "pixelated low-resolution, awkward childlike proportions, "
    "MS Paint doodle, deliberately badly drawn, "
    "a man holding a coffee cup, a small plant, a hanging light"
)
NEGATIVE = (
    "photorealistic, sharp, smooth, professional, high quality, "
    "anti-aliased, gradient, busy background, detailed scenery, "
    "cafe interior, brick wall, complex composition"
)


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    ref_path = USER_TEST / "set1_output.png"
    if not src_path.exists():
        print(f"missing {src_path}")
        return

    init = Image.open(src_path).convert("RGB")
    print(f"[match] input {init.size}")

    # Try TXT2IMG instead of img2img: with the user reference, the
    # ChatGPT model abstracts the cafe scene into 'man + coffee + plant'
    # on plain white paper. img2img keeps too much scenery context.
    # Pure txt2img with strong "isolated subjects on white" prompt
    # gives SD a chance to produce that diagram-like layout.
    print(f"[match] running SD-Turbo txt2img...")
    t = time.time()
    sd_out = generator.generate_txt2img(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        steps=4,
        guidance=0.0,
        width=512,
        height=512,
        seed=42,
    )
    print(f"[match] SD done in {time.time() - t:.1f}s")
    sd_path = SAMPLES / "match_set1_sd.png"
    sd_out.save(sd_path)

    # Apply kasun_color with maximally aggressive abstraction: kill all
    # internal SD detail with heavy median, allow more palette slots, and
    # raise the line threshold so only true region boundaries survive.
    kasun_color_out = kasun_color(
        sd_out,
        grid=96,
        colors=8,
        smooth=15,           # very heavy median = only dominant regions survive
        saturation=1.8,
        line_threshold=120,  # only strong boundaries, kills internal noise
        dilate=0,
    )
    kasun_color_path = SAMPLES / "match_set1_kasun_color.png"
    kasun_color_out.save(kasun_color_path)

    # Also kasun_line for comparison
    kasun_line_out = kasun_line(sd_out)
    kasun_line_path = SAMPLES / "match_set1_kasun_line.png"
    kasun_line_out.save(kasun_line_path)

    # Index
    md = [
        "# kasun match attempt vs user ChatGPT reference",
        "",
        "Pipeline: photo -> SD-Turbo img2img (doodle prompt) -> kasun.",
        "",
        "| stage | image |",
        "| --- | --- |",
        f"| input photo | ![](reference/user_test/line/set1_input.png) |",
        f"| SD-Turbo img2img | ![](match_set1_sd.png) |",
        f"| kasun_color | ![](match_set1_kasun_color.png) |",
        f"| kasun_line | ![](match_set1_kasun_line.png) |",
        f"| **ChatGPT reference (target)** | ![](reference/user_test/line/set1_output.png) |",
    ]
    (SAMPLES / "match_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[match] index -> samples/match_grid.md")


if __name__ == "__main__":
    main()
