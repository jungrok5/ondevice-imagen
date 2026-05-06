"""Max-quality showcase on the current hardware (AMD Radeon Pro 580X, CPU torch).

Generates two reference images using popular SD 1.5 fine-tunes:
  - samples/quality_realistic.png  via Lykon/dreamshaper-8        (realistic)
  - samples/quality_anime.png      via Linaqruf/anything-v3-better-vae (anime)

Both run at full quality (30 steps, DPMSolver, guidance 7) — the opposite end
of the speed/quality dial from SD-Turbo. Each image takes 1-3 minutes on CPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

PROMPT = (
    "a young woman holding a coffee cup at a window-side cafe, "
    "soft afternoon sunlight, plants in background, detailed eyes, "
    "shallow depth of field"
)

NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra fingers, "
    "blurry, jpeg artifacts, watermark, text, deformed, ugly, low quality"
)


RUNS = [
    {
        "name": "realistic",
        "model": "Lykon/dreamshaper-8",
        "out": SAMPLES / "quality_realistic.png",
    },
    {
        "name": "anime",
        "model": "Linaqruf/anything-v3-better-vae",
        "out": SAMPLES / "quality_anime.png",
    },
]


def run(run_cfg: dict) -> None:
    print(f"\n[quality] === {run_cfg['name']}: {run_cfg['model']} ===")
    t_load = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        run_cfg["model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    print(f"[quality] loaded in {time.time() - t_load:.1f}s")

    t_gen = time.time()
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=30,
        guidance_scale=7.0,
        width=512,
        height=512,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    img = out.images[0]
    img.save(run_cfg["out"])
    print(f"[quality] generated in {time.time() - t_gen:.1f}s -> {run_cfg['out']}")

    # free memory before next model
    del pipe
    import gc

    gc.collect()


if __name__ == "__main__":
    print(f"[quality] prompt : {PROMPT}")
    for cfg in RUNS:
        run(cfg)
    print("\n[quality] all done")
