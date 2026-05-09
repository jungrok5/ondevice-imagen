"""Phase 2 Step 3 verify — regenerate e2 cell with fp16 ONNX pipe and
visually compare to Step 2's fp32 ONNX cell. Same numpy seed so any
difference is purely the fp32 → fp16 weight precision drop, isolated
from RNG differences.

Expected outcome: small per-pixel numerical drift, same overall
aesthetic (pencil + partial color + cafe/rain prompt). Anything that
visibly degrades (banding, noise blowups, washed colors) means fp16
isn't safe for this LoRA and we'd fall back to selectively keeping
some sub-models at fp32.
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

ONNX_DIR = ROOT / "models" / "onnx_fp16" / "sd15_drawing_nty_scale0.8"
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

VERIFY_PNG = SAMPLES / f"phase2_step3_onnx_fp16_verify_{VERIFY_EVENT_TAG}.png"
STEP2_PNG  = SAMPLES / f"phase2_step2_onnx_verify_{VERIFY_EVENT_TAG}.png"


def main() -> None:
    print(f"[fp16-verify] loading fp16 ONNX pipe from {ONNX_DIR}")
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        str(ONNX_DIR), provider="CPUExecutionProvider",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )
    print(f"[fp16-verify] generating cell {VERIFY_EVENT_TAG} ...")

    built = build_event_prompt(
        VERIFY_EVENT, visual_style="sd15_drawing_nty", prompt_seed=PROMPT_SEED,
    )
    print(f"[fp16-verify]   prompt: {built.positive[:120]}...")

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
    print(f"[fp16-verify]   {time.time() - t:.1f}s -> {VERIFY_PNG.name}")

    if STEP2_PNG.exists():
        from PIL import Image, ImageChops
        a = Image.open(STEP2_PNG).convert("RGB")
        b = Image.open(VERIFY_PNG).convert("RGB")
        diff = ImageChops.difference(a, b)
        bbox = diff.getbbox()
        hist = diff.histogram()
        zero_per_ch = (hist[0], hist[256], hist[512])
        total_per_ch = a.size[0] * a.size[1]
        nonzero_per_ch = tuple(total_per_ch - z for z in zero_per_ch)
        # Mean abs diff per channel
        mean_abs = []
        for ch in range(3):
            s = sum(i * hist[ch * 256 + i] for i in range(256))
            mean_abs.append(s / total_per_ch)
        print()
        print(f"[fp16-verify] step2 (fp32) vs step3 (fp16) pixel diff:")
        print(f"  bbox: {bbox}")
        print(f"  per-channel non-zero pixels (R,G,B): {nonzero_per_ch} / {total_per_ch}")
        print(f"  per-channel mean abs diff (R,G,B): {tuple(round(x, 2) for x in mean_abs)} / 255")


if __name__ == "__main__":
    main()
