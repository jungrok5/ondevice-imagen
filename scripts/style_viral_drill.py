"""Round 2 of viral-aesthetic search: drill into the two strongest finds.

Round 1 (style_viral_search.py) flagged friends_mom_portrait and
hotel_art_70s as the two presets with real viral DNA. This round
explores variants of each — different cultural rhymes / demographics /
settings — to see which sub-direction has the strongest hook.

Same subject + seed as round 1 so the new images line up against
samples/viral_friends_mom_portrait.png and samples/viral_hotel_art_70s.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import styles  # noqa: E402

SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


SUBJECT = (
    "a young woman holding a coffee cup at a cafe window, plants beside her"
)


PRESETS_TO_RUN = [
    # friends_mom variants
    styles.MOM_KOREAN_LIVING_ROOM,
    styles.RETIRED_ENGINEER_WATERCOLOR,
    # hotel_art variants
    styles.KOREAN_WEDDING_HALL_90S,
    styles.DOCTORS_OFFICE_85,
]


MODEL = "dreamlike-art/dreamlike-diffusion-1.0"
SEED = 7
STEPS = 30


def main() -> None:
    print(f"[drill] subject : {SUBJECT}")
    print(f"[drill] model   : {MODEL}\n")

    print(f"[drill] loading {MODEL}")
    t = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    print(f"[drill] loaded in {time.time() - t:.1f}s")

    for preset in PRESETS_TO_RUN:
        out_path = SAMPLES / f"drill_{preset.name}.png"
        if out_path.exists():
            print(f"[drill] {preset.name}: skipped (exists)")
            continue

        positive, negative = styles.build_prompt(preset, SUBJECT)
        print(f"\n[drill] === {preset.name} ===")
        print(f"[drill] prompt: {positive[:140]}...")

        t = time.time()
        out = pipe(
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        out.images[0].save(out_path)
        print(f"[drill] {preset.name} done in {time.time() - t:.1f}s -> {out_path}")

    print("\n[drill] all done")


if __name__ == "__main__":
    main()
