"""Phase 3 Step 1 — bake the PC-verified SDXL LoRA into SDXL-Turbo.

Phase 2 put SD 1.5 + sd15_drawing on the phone. But the styles we
actually chose in the README grids (worstimever, mspaint_portraits)
are *SDXL* LoRAs running on SDXL-Turbo — a generation above what the
phone currently renders. Phase 3 closes that gap: carry the exact
PC-verified SDXL-Turbo + LoRA combo to Android via int8.

Why Turbo helps mobile beyond quality:
  - guidance_scale=0.0 → no CFG → UNet batch=1 (Phase 2 ran batch=2)
  - 4 steps vs 12 → with batch, 6x fewer UNet forwards than Phase 2
    before any quantization / NNAPI work even starts.

Same rationale as Phase 2 Step 1: ONNX is a static format, no LoRA
hot-swap — ship ONE fused checkpoint per visual mode.

Output: diffusers-format dir at
    models/fused_xl/sdxl_turbo_worstimever_scale0.9/
for Step 2's optimum ONNX export (export_sdxl_onnx_phase3.py).

Verification: the SDXL grids were generated with the non-deterministic
prompt builder (no prompt_seed), so unlike Phase 2 we can't pixel-diff
against a committed grid cell. Instead we fix prompt_seed=7 and render
the verify cell from the fused fp32 pipe before saving — fuse_lora
determinism itself was already proven in Phase 2 (pixel-diff bbox =
None). Eyeball the PNG for the worstimever register (thick scratchy
outlines, saturated cartoon).
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
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt  # noqa: E402

LORA_DIR = ROOT / "models" / "lora"
FUSED_ROOT = ROOT / "models" / "fused_xl"
SAMPLES = ROOT / "samples"

# === knobs (mirror compare_event_variations.py exactly) ===
BASE_MODEL = "stabilityai/sdxl-turbo"
LORA_FILE  = "worstimever_xl.safetensors"   # or sdxl_mspaint_portraits.safetensors
LORA_LABEL = "worstimever"
LORA_SCALE = 0.9

DIFFUSION_SEED = 42
PROMPT_SEED    = 7      # grids were random; verify cell pins the phrase pick
STEPS          = 4      # Turbo tier
CFG            = 0.0    # Turbo: no CFG → phone UNet runs batch=1
WIDTH          = 512
HEIGHT         = 512

VERIFY_EVENT_TAG = "e1_xmas_evening"
VERIFY_EVENT = {
    "time": "19:00", "weather": "맑음", "date": "2026-12-25",
    "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
}

OUT_DIR = FUSED_ROOT / f"sdxl_turbo_{LORA_LABEL}_scale{LORA_SCALE}"
VERIFY_PNG = SAMPLES / f"phase3_step1_fused_verify_{VERIFY_EVENT_TAG}.png"


def main() -> None:
    print(f"[merge-xl] base={BASE_MODEL} lora={LORA_FILE} scale={LORA_SCALE}")
    print(f"[merge-xl] output dir: {OUT_DIR}")
    print(f"[merge-xl] verify PNG: {VERIFY_PNG}")
    print()

    # fp32 merge + verify (same policy as Phase 2: confirm the merge at
    # full precision, cast to fp16 only at save time).
    print("[merge-xl] loading SDXL-Turbo in fp32...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )

    print(f"[merge-xl] loading + fusing LoRA at scale {LORA_SCALE}...")
    pipe.load_lora_weights(
        str(LORA_DIR), weight_name=LORA_FILE, adapter_name=LORA_LABEL,
    )
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    pipe.unload_lora_weights()

    pipe = pipe.to("cpu")

    # === Verify ===
    print(f"\n[merge-xl] verify: rendering {VERIFY_EVENT_TAG} ...")
    built = build_event_prompt(
        VERIFY_EVENT, visual_style=LORA_LABEL, prompt_seed=PROMPT_SEED,
    )
    print(f"[merge-xl]   prompt: {built.positive[:120]}...")

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
    print(f"[merge-xl]   {time.time() - t:.1f}s -> {VERIFY_PNG.name}")

    # === Save fused checkpoint as fp16 ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[merge-xl] casting to fp16 + saving to {OUT_DIR} ...")
    pipe.unet           = pipe.unet.to(torch.float16)
    pipe.text_encoder   = pipe.text_encoder.to(torch.float16)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.float16)
    pipe.vae            = pipe.vae.to(torch.float16)
    pipe.save_pretrained(str(OUT_DIR), safe_serialization=True)

    total = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"[merge-xl] fused checkpoint saved ({total / 1_000_000_000:.2f} GB on disk)")

    print(
        f"\n[merge-xl] DONE. Next: python scripts/export_sdxl_onnx_phase3.py"
        f"  (exports {OUT_DIR.name} to ONNX)"
    )
    print(
        "[merge-xl] eyeball the verify PNG for the worstimever register "
        "(thick scratchy outlines, saturated cartoon, xmas tree + dusk)."
    )


if __name__ == "__main__":
    main()
