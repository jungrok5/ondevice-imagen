"""Bare LoRA-capability test — what does each LoRA *actually* produce?

No STYLE_TAGS prefix. No NEGATIVE prompt. No translation pipeline.
Just: LoRA-card trigger + minimal English subject.

This isolates "what does the LoRA do natively?" from "what does our
custom prompt prefix do?". Result tells us:
  - Which LoRAs are pure style (apply to any subject the same way)
  - Which LoRAs are subject-leaning (drag the output toward their
    training distribution regardless of the requested subject)
  - How strong each LoRA's signature looks at default scale 0.9

We use SDXL-Turbo as the base for cleanest LoRA expression (single
distilled UNet, no Lightning-LoRA-vs-style-LoRA conflict). It is
non-commercial-licensed; in product the same LoRAs would ship with
SDXL-Lightning-UNet variant (option A from the earlier shootout).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"
LORA_DIR = ROOT / "models" / "lora"

# Single common subject across every cell — the ONLY thing varying
# row-to-row is the LoRA + its trigger.
SUBJECT = "a young man holding a coffee cup at a cafe table"

# (display name, file [None = no LoRA], trigger to PREPEND)
ROWS = [
    ("00_baseline_no_lora",  None,                                  ""),
    ("01_worstimever",       "worstimever_xl.safetensors",           "DD-wte artstyle, "),
    ("02_mspaint_portraits", "sdxl_mspaint_portraits.safetensors",   "MSPaint drawing of "),
    ("03_pixel_art_xl",      "pixel_art_xl.safetensors",             "pixel art, "),
    ("04_lah_cute_social",   "lah_cute_social.safetensors",          "cute doodle, "),
]

SEED = 42


def main() -> None:
    print(f"[raw] subject: {SUBJECT}\n")

    print("[raw] loading SDXL-Turbo (clean baseline)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    md = [
        "# Raw LoRA capability — trigger + subject only",
        "",
        "No STYLE_TAGS prefix, no NEGATIVE prompt, no translated phrase",
        "blocks. Just `{LoRA trigger}, {subject}`. Same subject across",
        "all rows; same seed (42); base SDXL-Turbo.",
        "",
        f"Subject: `{SUBJECT}`",
        "",
        "| LoRA | trigger | full prompt | result |",
        "| --- | --- | --- | --- |",
    ]

    for name, fname, trigger in ROWS:
        # Reset
        try: pipe.unfuse_lora()
        except Exception: pass
        try: pipe.unload_lora_weights()
        except Exception: pass

        if fname:
            pipe.load_lora_weights(str(LORA_DIR), weight_name=fname, adapter_name=name)
            pipe.fuse_lora(lora_scale=0.9)

        prompt = f"{trigger}{SUBJECT}"
        print(f"\n[raw] === {name} ===")
        print(f"[raw] prompt: {prompt}")

        t = time.time()
        out = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        img = out.images[0]
        img_path = SAMPLES / f"raw_lora_{name}.png"
        img.save(img_path)
        print(f"[raw]   {time.time() - t:.1f}s -> {img_path.name}")

        trigger_disp = trigger.rstrip(", ") if trigger else "(none)"
        md.append(
            f"| `{name}` | `{trigger_disp}` | `{prompt}` | ![]({img_path.name}) |"
        )

    (SAMPLES / "raw_lora_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[raw] index -> samples/raw_lora_grid.md")


if __name__ == "__main__":
    main()
