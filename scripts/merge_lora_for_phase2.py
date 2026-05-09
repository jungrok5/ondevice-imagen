"""Phase 2 Step 1 — bake the chosen LoRA into the SD 1.5 base.

Why this exists: Core ML (.mlpackage), MediaPipe (.task), and ONNX
Runtime Mobile (.onnx) are all *static* model formats — no LoRA
hot-swap interface. The mobile pipeline ships ONE base+LoRA
checkpoint already merged. This script does that merge on PC, where
diffusers can do `load_lora_weights -> fuse_lora -> save_pretrained`.

Output: a diffusers-format directory at
    models/fused/sd15_drawing_nty_scale0.8/
suitable for `optimum-cli export onnx --model <dir>` in Step 2.

Verification: regenerates one cell from the SD 1.5 grid (e2_evening_cafe_rain)
using the merged pipe and saves it next to the originals so we can
diff visually. With identical seed + prompt + scheduler + steps,
the merged output should match the grid cell pixel-for-pixel modulo
fp32 -> fp16 rounding (we keep verification at fp32 to isolate the
merge step from the precision step).

Saved fused dir is fp16 so Step 2's ONNX export stays under ~2 GB.
The fp32 PyTorch run on CPU is just for the verify PNG and gets
discarded.
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

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt  # noqa: E402

LORA_DIR = ROOT / "models" / "lora"
FUSED_ROOT = ROOT / "models" / "fused"
SAMPLES = ROOT / "samples"

# === knobs (mirror sd15_lora_compare.py exactly so verify cell matches) ===
BASE_MODEL    = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_FILE     = "sd15_drawing_nty.safetensors"
LORA_LABEL    = "sd15_drawing_nty"
LORA_SCALE    = 0.8

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

OUT_DIR = FUSED_ROOT / f"{LORA_LABEL}_scale{LORA_SCALE}"
VERIFY_PNG = SAMPLES / f"phase2_step1_fused_verify_{VERIFY_EVENT_TAG}.png"
GRID_PNG   = SAMPLES / f"sd15_compare_{LORA_LABEL}__{VERIFY_EVENT_TAG}.png"


def main() -> None:
    print(f"[merge] base={BASE_MODEL} lora={LORA_FILE} scale={LORA_SCALE}")
    print(f"[merge] output dir: {OUT_DIR}")
    print(f"[merge] verify PNG: {VERIFY_PNG}")
    print(f"[merge] grid cell to compare against: {GRID_PNG}")
    print()

    # fp32 for the merge + verify, so the verify cell is bit-identical to
    # the grid cell (which was generated at fp32). fp16 cast happens AT
    # save time, after we've confirmed the merge is correct.
    print("[merge] loading base in fp32...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )

    print(f"[merge] loading + fusing LoRA at scale {LORA_SCALE}...")
    pipe.load_lora_weights(
        str(LORA_DIR), weight_name=LORA_FILE, adapter_name=LORA_LABEL,
    )
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    # Drop the adapter object — its weights are baked into the model now.
    # save_pretrained refuses to serialize a pipe that still has loaded
    # adapters anyway.
    pipe.unload_lora_weights()

    pipe = pipe.to("cpu")

    # === Verify ===
    print(f"\n[merge] verify: regenerating cell {VERIFY_EVENT_TAG} ...")
    built = build_event_prompt(
        VERIFY_EVENT, visual_style=LORA_LABEL, prompt_seed=PROMPT_SEED,
    )
    print(f"[merge]   prompt: {built.positive[:120]}...")

    t = time.time()
    img = pipe(
        prompt=built.positive,
        negative_prompt=built.negative,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        width=WIDTH,
        height=HEIGHT,
        generator=torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED),
    ).images[0]
    img.save(VERIFY_PNG)
    print(f"[merge]   {time.time() - t:.1f}s -> {VERIFY_PNG.name}")

    # === Save fused checkpoint as fp16 ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[merge] casting to fp16 + saving to {OUT_DIR} ...")
    pipe.unet      = pipe.unet.to(torch.float16)
    pipe.text_encoder = pipe.text_encoder.to(torch.float16)
    pipe.vae       = pipe.vae.to(torch.float16)
    pipe.save_pretrained(str(OUT_DIR), safe_serialization=True)

    # Sanity: total dir size on disk
    total = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"[merge] fused checkpoint saved ({total / 1_000_000_000:.2f} GB on disk)")

    print(
        f"\n[merge] DONE. Next step: optimum-cli export onnx "
        f"--model {OUT_DIR} models/onnx/{OUT_DIR.name}/"
    )
    print(
        f"[merge] visually diff {VERIFY_PNG.name} vs {GRID_PNG.name} "
        f"— should be near-identical at fp32 (fuse step has no precision loss)."
    )


if __name__ == "__main__":
    main()
