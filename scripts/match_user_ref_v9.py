"""v9 — SD-Turbo + COOLKIDS LoRA, NO jitter (user feedback: jitter hurts).

LoRA path. COOLKIDS V2 (Clumsy_Trainer, SD 1.5 compatible) trains the
naive children-book illustration register directly into the model
weights, so we shouldn't need such heavy post-processing to fake it.

Pipeline (jitter removed per user note that v7/v8 jittered outputs
felt off — the clean SD output reads cleaner):
  photo
   -> SD-Turbo img2img (LoRA loaded, cartoon prompt)
   -> cv2.stylization (optional, light)
   -> mild desaturate

Compares:
  - LoRA at scale 0.6 (subtle infusion, identity preserved)
  - LoRA at scale 0.9 (stronger, more child-art register)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from diffusers import AutoPipelineForImage2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.match_user_ref_v7 import stylize, apply_jitter  # noqa: E402

SAMPLES = ROOT / "samples"
USER_TEST = SAMPLES / "reference" / "user_test" / "line"
LORA_PATH = ROOT / "models" / "lora" / "COOLKIDS.safetensors"


PROMPT = (
    "coolkids style, simple flat children illustration, naive figure, "
    "thin black outlines, pastel colour fill, awkward kid proportions"
)
NEGATIVE = (
    "photorealistic, detailed, sharp focus, oil painting, smooth gradient, "
    "professional, hd, vector art, pixel art, polished"
)
STRENGTH = 0.75
SEED = 42


def main() -> None:
    src_path = USER_TEST / "set1_input.png"
    photo = Image.open(src_path).convert("RGB")
    print(f"[v9] photo {photo.size}")

    # SD-Turbo is actually SD 2.1 based (cross-attn dim 1024) — incompatible
    # with COOLKIDS LoRA which is SD 1.5 (768 dim). Switch to SD 1.5 base.
    print(f"[v9] loading SD 1.5 + COOLKIDS LoRA...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    pipe.load_lora_weights(
        str(LORA_PATH.parent), weight_name=LORA_PATH.name
    )
    print(f"[v9] LoRA loaded.")

    md = [
        "# v9 — SD-Turbo + COOLKIDS LoRA + cv2 stylize + jitter",
        "",
        "Adds the COOLKIDS V2 LoRA (Clumsy_Trainer, SD 1.5 compatible,",
        "144 MB) so SD-Turbo natively produces children-book illustration",
        "without needing as heavy post-processing.",
        "",
        "| variant | image |",
        "| --- | --- |",
        f"| input | ![](reference/user_test/line/set1_input.png) |",
    ]

    for tag, scale in (("s06", 0.6), ("s09", 0.9)):
        print(f"\n[v9] LoRA scale {scale} ...")
        pipe.fuse_lora(lora_scale=scale)

        t = time.time()
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=photo,
            num_inference_steps=20,    # SD 1.5 base wants more steps
            guidance_scale=7.0,        # and proper guidance
            strength=STRENGTH,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        sd = out.images[0]
        print(f"[v9]    SD done in {time.time() - t:.1f}s")

        sd_path = SAMPLES / f"v9_{tag}_sd.png"
        sd.save(sd_path)

        # Light cv2 stylization for cleaner edges, NO jitter.
        stylized = stylize(sd, sigma_s=60, sigma_r=0.45)
        finished = ImageEnhance.Color(stylized).enhance(0.9)

        final_path = SAMPLES / f"v9_{tag}_final.png"
        finished.save(final_path)

        md.append(f"| LoRA scale={scale} (raw SD) | ![]({sd_path.name}) |")
        md.append(f"| LoRA scale={scale} + cv2 (no jitter) | ![]({final_path.name}) |")

        # Unfuse before next iteration
        pipe.unfuse_lora()

    md.append(f"| target — ChatGPT 하찮은 | ![](reference/user_test/line/set1_output.png) |")
    md.append(f"| user-favourite — v5 s=0.85 | ![](v5_set1_sd_s85.png) |")

    (SAMPLES / "match_v9_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[v9] index -> samples/match_v9_grid.md")


if __name__ == "__main__":
    main()
