"""Phase 3 Step 3a — capture real UNet inputs for int8 calibration.

QDQ static quantization (Step 3b, the NNAPI-friendly format) needs
calibration data: representative *real* UNet inputs, not random
tensors. Random latents miss the actual activation ranges — the
denoising trajectory starts at pure noise and sharpens, and Turbo's
distilled schedule covers a narrow sigma set (4 steps), so ranges
from real runs are tight and cheap to collect.

Method: monkey-patch the fused fp32 pipe's unet.forward to record
every call's inputs, then run the 5 README events through the normal
pipeline. 5 events x 4 steps = 20 samples — small but each is the
full (latent, timestep, prompt-embedding, micro-conditioning) tuple
at exactly the operating point the phone will run.

Output: models/calib_xl/<fused_name>/sample_XXX.npz with keys
    sample                  (1, 4, 64, 64)   latent at that step
    timestep                ()               int64
    encoder_hidden_states   (1, 77, 2048)    TE1(768) cat TE2(1280)
    text_embeds             (1, 1280)        TE2 pooled
    time_ids                (1, 6)           orig/crop/target size

These keys deliberately match the ONNX UNet's input names from the
optimum SDXL export so Step 3b's CalibrationDataReader can feed the
npz dicts straight in.
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
import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt  # noqa: E402

# === knobs (must mirror the Step 1 fuse + grid script) ===
FUSED_NAME = "sdxl_turbo_worstimever_scale0.9"
FUSED_DIR  = ROOT / "models" / "fused_xl" / FUSED_NAME
CALIB_DIR  = ROOT / "models" / "calib_xl" / FUSED_NAME
VISUAL_STYLE = "worstimever"

DIFFUSION_SEED = 42
PROMPT_SEED    = 7
STEPS          = 4
CFG            = 0.0
WIDTH          = 512
HEIGHT         = 512

# The 5 README grid events — the phone's actual input distribution.
EVENTS = [
    ("e1_xmas_evening", {
        "time": "19:00", "weather": "맑음", "date": "2026-12-25",
        "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
    }),
    ("e2_summer_park", {
        "time": "14:30", "weather": "맑음", "date": "2026-08-15",
        "country": "대한민국", "city": "서울시", "place": "한강 공원",
    }),
    ("e3_rainy_cafe", {
        "time": "08:45", "weather": "비", "date": "2026-03-20",
        "country": "대한민국", "city": "서울시", "place": "광화문 카페",
    }),
    ("e4_autumn_night_home", {
        "time": "23:30", "weather": "흐림", "date": "2026-10-31",
        "country": "대한민국", "city": "서울시", "place": "무궁화 아파트",
    }),
    ("e5_snow_noon_market", {
        "time": "12:00", "weather": "눈", "date": "2026-02-04",
        "country": "대한민국", "city": "서울시", "place": "광장시장",
    }),
]


def main() -> None:
    if not FUSED_DIR.exists():
        raise SystemExit(f"missing fused dir — run Step 1 first: {FUSED_DIR}")
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[calib] loading fused fp32 pipe from {FUSED_DIR.name} ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        FUSED_DIR, torch_dtype=torch.float32,
    ).to("cpu")

    captured: list[dict] = []
    real_forward = pipe.unet.forward

    def recording_forward(sample, timestep, encoder_hidden_states,
                          *args, **kwargs):
        added = kwargs.get("added_cond_kwargs") or {}
        captured.append({
            "sample": sample.detach().cpu().float().numpy(),
            "timestep": np.asarray(
                int(timestep.item() if hasattr(timestep, "item") else timestep),
                dtype=np.int64,
            ),
            "encoder_hidden_states":
                encoder_hidden_states.detach().cpu().float().numpy(),
            "text_embeds": added["text_embeds"].detach().cpu().float().numpy(),
            "time_ids": added["time_ids"].detach().cpu().float().numpy(),
        })
        return real_forward(sample, timestep, encoder_hidden_states,
                            *args, **kwargs)

    pipe.unet.forward = recording_forward

    for tag, ev in EVENTS:
        built = build_event_prompt(
            ev, visual_style=VISUAL_STYLE, prompt_seed=PROMPT_SEED,
        )
        print(f"[calib] {tag}: {built.positive[:90]}...")
        t = time.time()
        pipe(
            prompt=built.positive,
            negative_prompt=built.negative,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            width=WIDTH,
            height=HEIGHT,
            generator=torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED),
        )
        print(f"[calib]   {time.time() - t:.1f}s, {len(captured)} samples so far")

    pipe.unet.forward = real_forward

    for i, rec in enumerate(captured):
        np.savez(CALIB_DIR / f"sample_{i:03d}.npz", **rec)

    total = sum(f.stat().st_size for f in CALIB_DIR.glob("*.npz"))
    print(
        f"\n[calib] DONE — {len(captured)} samples "
        f"({total / 1_000_000:.0f} MB) in {CALIB_DIR}"
    )
    print("[calib] next: python scripts/quantize_sdxl_unet_int8_phase3.py")


if __name__ == "__main__":
    main()
