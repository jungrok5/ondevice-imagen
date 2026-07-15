"""Phase 3 Step 2 — ONNX export of the fused SDXL-Turbo checkpoint.

Phase 2 used a patched copy of the official diffusers SD-1.5 converter
(_convert_sd_to_onnx.py). That script has no SDXL support — SDXL adds
a second text encoder and the micro-conditioning inputs (text_embeds,
time_ids), so the export graph differs. The maintained path for SDXL
is HuggingFace optimum:

    pip install "optimum[exporters]" onnx onnxruntime

We call optimum's main_export() directly instead of shelling out to
optimum-cli so the knobs live in this file like every other script.

Output layout mirrors Phase 2 (models/onnx_xl/<name>/<submodel>/model.onnx):
    text_encoder/    CLIP ViT-L      (768-dim hidden states)
    text_encoder_2/  OpenCLIP bigG   (1280-dim + pooled text_embeds)
    unet/            2.6B params — the int8 target for Step 3
    vae_encoder/ vae_decoder/
    tokenizer/ tokenizer_2/ scheduler/ model_index.json

Expect ~10 GB fp32 on disk (UNet alone ~10x SD1.5's 3.3 GB at fp32 is
wrong — it's ~2.6B params ≈ 10.3 GB fp32, hence Step 3's int8 is not
optional this time; fp32 SDXL will not fit an 8 GB phone at all).

Optional verify: if optimum's ORT pipeline class is importable we
regenerate Step 1's verify cell through ONNX Runtime CPU and save it
next to the fp32 PyTorch one for a visual diff (same policy as
Phase 2 Step 2's verify cell).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# === knobs ===
FUSED_NAME = "sdxl_turbo_worstimever_scale0.9"     # Step 1 output
FUSED_DIR  = ROOT / "models" / "fused_xl" / FUSED_NAME
ONNX_DIR   = ROOT / "models" / "onnx_xl" / FUSED_NAME

DIFFUSION_SEED = 42
PROMPT_SEED    = 7
STEPS          = 4
CFG            = 0.0
WIDTH          = 512
HEIGHT         = 512

VERIFY_EVENT_TAG = "e1_xmas_evening"
VERIFY_EVENT = {
    "time": "19:00", "weather": "맑음", "date": "2026-12-25",
    "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
}
VERIFY_PNG = ROOT / "samples" / f"phase3_step2_onnx_verify_{VERIFY_EVENT_TAG}.png"


def size_report(root: Path) -> None:
    total = 0
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sz = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
        total += sz
        print(f"  {sub.name:20s} {sz / 1_000_000:9.1f} MB")
    print(f"  {'TOTAL':20s} {total / 1_000_000_000:9.2f} GB")


def main() -> None:
    if not FUSED_DIR.exists():
        raise SystemExit(
            f"missing fused dir — run Step 1 first "
            f"(scripts/merge_sdxl_lora_phase3.py): {FUSED_DIR}"
        )

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        raise SystemExit(
            "optimum not installed. This step needs:\n"
            '    pip install "optimum[exporters]" onnx onnxruntime\n'
            "then re-run. (Kept out of requirements.txt — PC export "
            "stack only, same policy as Phase 2's onnx deps.)"
        )

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[onnx-xl] exporting {FUSED_DIR.name} -> {ONNX_DIR}")
    print("[onnx-xl] task=stable-diffusion-xl, fp32 opset default")
    t = time.time()
    main_export(
        model_name_or_path=str(FUSED_DIR),
        output=str(ONNX_DIR),
        task="stable-diffusion-xl",
        # fp32 export, same policy as Phase 2 Step 2: precision work is a
        # separate, verifiable step (int8 in Step 3), not folded into export.
    )
    print(f"[onnx-xl] exported in {time.time() - t:.0f}s")
    size_report(ONNX_DIR)

    # === optional verify through ORT CPU ===
    try:
        from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    except ImportError:
        print(
            "\n[onnx-xl] optimum.onnxruntime pipeline not available — "
            "skipping the ORT verify cell (export itself is done). "
            "Install onnxruntime + optimum[onnxruntime] to enable."
        )
        return

    from src.event_prompt import build_event_prompt  # noqa: E402
    import numpy as np  # noqa: F401  (ORT pipelines pull numpy anyway)

    print("\n[onnx-xl] verify: regenerating Step 1 cell through ORT CPU ...")
    pipe = ORTStableDiffusionXLPipeline.from_pretrained(str(ONNX_DIR))
    built = build_event_prompt(
        VERIFY_EVENT, visual_style="worstimever", prompt_seed=PROMPT_SEED,
    )
    import torch
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
    print(f"[onnx-xl]   {time.time() - t:.1f}s -> {VERIFY_PNG.name}")
    print(
        "[onnx-xl] visually diff against phase3_step1_fused_verify — "
        "should match modulo fp32 PyTorch vs ORT kernel rounding."
    )


if __name__ == "__main__":
    main()
