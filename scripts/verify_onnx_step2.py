"""Phase 2 Step 2 verify — regenerate the e2_evening_cafe_rain cell with
the ONNX-exported pipe and pixel-diff against Step 1's fp32 verify PNG.

ONNX inference uses graph execution + ONNX-mathematical scheduling
which can produce small numerical drift vs PyTorch eager execution
even at the same dtype. Visual diff should still be near-identical
(SSIM > 0.98 typical).

Run AFTER `optimum-cli export onnx ... models/onnx/sd15_drawing_nty_scale0.8/`
finishes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
from diffusers import OnnxStableDiffusionPipeline, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt  # noqa: E402

ONNX_DIR = ROOT / "models" / "onnx" / "sd15_drawing_nty_scale0.8"
SAMPLES = ROOT / "samples"

DIFFUSION_SEED = 42
PROMPT_SEED   = 7
STEPS         = 12
CFG           = 7.0
WIDTH         = 512
HEIGHT        = 512

VERIFY_EVENT_TAG = "e2_evening_cafe_rain"
VERIFY_EVENT = {
    "time": "20:00", "weather": "비", "date": "2026-10-08",
    "country": "대한민국", "city": "서울시", "place": "광화문 카페",
}

VERIFY_PNG = SAMPLES / f"phase2_step2_onnx_verify_{VERIFY_EVENT_TAG}.png"
STEP1_PNG  = SAMPLES / f"phase2_step1_fused_verify_{VERIFY_EVENT_TAG}.png"


def main() -> None:
    print(f"[onnx-verify] loading ONNX pipe from {ONNX_DIR}")
    pipe = OnnxStableDiffusionPipeline.from_pretrained(str(ONNX_DIR), provider="CPUExecutionProvider")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )
    print(f"[onnx-verify] generating cell {VERIFY_EVENT_TAG} ...")

    built = build_event_prompt(
        VERIFY_EVENT, visual_style="sd15_drawing_nty", prompt_seed=PROMPT_SEED,
    )
    print(f"[onnx-verify]   prompt: {built.positive[:120]}...")

    t = time.time()
    img = pipe(
        prompt=built.positive,
        negative_prompt=built.negative,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        width=WIDTH,
        height=HEIGHT,
        generator=np.random.RandomState(DIFFUSION_SEED),
    ).images[0]
    img.save(VERIFY_PNG)
    print(f"[onnx-verify]   {time.time() - t:.1f}s -> {VERIFY_PNG.name}")

    # Pixel diff vs Step 1
    if STEP1_PNG.exists():
        from PIL import Image, ImageChops
        a = Image.open(STEP1_PNG).convert("RGB")
        b = Image.open(VERIFY_PNG).convert("RGB")
        diff = ImageChops.difference(a, b)
        bbox = diff.getbbox()
        hist = diff.histogram()
        # PIL histogram for an RGB image returns a 768-len list
        # (256*3) — index 0,256,512 are zero-diff per channel.
        zero_per_ch = (hist[0], hist[256], hist[512])
        total_per_ch = a.size[0] * a.size[1]
        nonzero_per_ch = tuple(total_per_ch - z for z in zero_per_ch)
        max_diff = max(
            (i for i in range(256) if hist[i] > 0 or hist[256+i] > 0 or hist[512+i] > 0),
            default=0,
        )
        print()
        print(f"[onnx-verify] step1 vs step2 pixel diff:")
        print(f"  bbox: {bbox}")
        print(f"  per-channel non-zero pixels (R,G,B): {nonzero_per_ch} / {total_per_ch}")
        print(f"  max abs diff (any channel): {max_diff} / 255")


if __name__ == "__main__":
    main()
