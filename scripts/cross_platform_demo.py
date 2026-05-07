"""Showcase: SD 1.5 fine-tunes that have *both* iOS and Android paths
without us writing any conversion code.

Each entry below has:
  - A pre-converted Core ML .mlpackage on Hugging Face (no work for iOS).
  - An ONNX export available via `optimum-cli export onnx --model <id> ...`
    that ONNX Runtime Mobile can run directly on Android (one CLI command).

Each generates a 512x512 PNG with the same neutral prompt + seed so the
only varying axis is the model. Each takes ~4 min on CPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

# Neutral prompt — picked so different fine-tunes have something to differ on.
PROMPT = (
    "a calm kitchen with morning light, plants on the windowsill, "
    "a ceramic mug of coffee on a wooden table, a folded newspaper, "
    "soft shadows, painterly atmosphere"
)
NEGATIVE = (
    "lowres, blurry, jpeg artifacts, watermark, text, deformed, "
    "bad anatomy, ugly, low quality"
)
SEED = 7
STEPS = 30


# Each tuple: (display name, HF id used for PC inference, iOS coreml repo, ONNX note)
CROSS_PLATFORM_MODELS = [
    {
        "name": "sd_1_5_base",
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "ios_coreml": "apple/coreml-stable-diffusion-v1-5 (Apple official)",
        "android_onnx": "optimum-cli export onnx",
    },
    {
        "name": "dreamlike_diffusion",
        "hf_id": "dreamlike-art/dreamlike-diffusion-1.0",
        "ios_coreml": "coreml-community/coreml-dreamlike-diffusion",
        "android_onnx": "optimum-cli export onnx",
    },
    {
        "name": "openjourney",
        "hf_id": "prompthero/openjourney",
        "ios_coreml": "coreml-community has openjourney conversions",
        "android_onnx": "optimum-cli export onnx",
    },
    {
        "name": "dreamlike_photoreal_2",
        "hf_id": "dreamlike-art/dreamlike-photoreal-2.0",
        "ios_coreml": "coreml-community has photoreal-2.0",
        "android_onnx": "optimum-cli export onnx",
    },
]


def run(cfg: dict) -> None:
    out_path = SAMPLES / f"cross_{cfg['name']}.png"
    if out_path.exists():
        print(f"\n[cross] === {cfg['name']}: skipped (exists) ===")
        return

    print(f"\n[cross] === {cfg['name']}: {cfg['hf_id']} ===")
    t = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["hf_id"],
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    print(f"[cross] loaded in {time.time() - t:.1f}s")

    t = time.time()
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=7.0,
        width=512,
        height=512,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    )
    out.images[0].save(out_path)
    print(f"[cross] {cfg['name']} generated in {time.time() - t:.1f}s -> {out_path}")

    del pipe
    import gc
    gc.collect()


if __name__ == "__main__":
    print(f"[cross] prompt: {PROMPT}\n")
    for cfg in CROSS_PLATFORM_MODELS:
        try:
            run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[cross] {cfg['name']} FAILED: {type(e).__name__}: {e}")
    print("\n[cross] all done")
