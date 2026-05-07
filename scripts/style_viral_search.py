"""Search for the next viral aesthetic on top of a painterly base model.

The 'pathetic prompt' (하찮은 프롬프트) trend went global in May 2025
because it had specific viral DNA: a meta-instruction to be deliberately
inept, recognizable-but-wrong output, earnest amateur energy, one-prompt
reusability, and a strong cultural reference (MS Paint).

This script tries five candidate aesthetics that share that DNA but
explore different cultural rhymes — to see whether SD 1.5 fine-tunes
can deliver any of them well enough to ride the same wave.

Uses dreamlike-diffusion-1.0 because it produced the most painterly /
artistic voice in the cross-platform comparison. Same neutral subject
and seed across all five so only the style varies.
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
    styles.RENAISSANCE_OIL,
    styles.HOTEL_ART_70S,
    styles.EMBROIDERED_SAMPLER,
    styles.FRIENDS_MOM_PORTRAIT,
    styles.HALF_REMEMBERED_FILM,
]


MODEL = "dreamlike-art/dreamlike-diffusion-1.0"
SEED = 7
STEPS = 30


def main() -> None:
    print(f"[viral] subject : {SUBJECT}")
    print(f"[viral] model   : {MODEL}\n")

    print(f"[viral] loading {MODEL}")
    t = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    print(f"[viral] loaded in {time.time() - t:.1f}s")

    for preset in PRESETS_TO_RUN:
        out_path = SAMPLES / f"viral_{preset.name}.png"
        if out_path.exists():
            print(f"[viral] {preset.name}: skipped (exists)")
            continue

        positive, negative = styles.build_prompt(preset, SUBJECT)
        print(f"\n[viral] === {preset.name} ===")
        print(f"[viral] prompt: {positive[:140]}...")

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
        print(f"[viral] {preset.name} done in {time.time() - t:.1f}s -> {out_path}")

    print("\n[viral] all done")


if __name__ == "__main__":
    main()
