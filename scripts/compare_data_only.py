"""Absolute baseline — just the translated event data, no extra prompt.

User question: 'what comes out if you just say -- draw with the info
I gave you'? Strip everything we previously layered on top:
  no STYLE_TAGS prefix
  no NEGATIVE prompt
  no LoRA trigger phrase
  no LoRA fused

Then add the same prompt to each of the two style LoRAs we like
(worstimever, mspaint_portraits) at default scale 0.9 — only the LoRA
fusion changes, not the prompt — to see what each LoRA adds on top of
that bare baseline.

Three cells:
  1. data only  + no LoRA
  2. data only  + worstimever
  3. data only  + mspaint_portraits
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_prompt import build_event_prompt  # noqa: E402

SAMPLES = ROOT / "samples"
LORA_DIR = ROOT / "models" / "lora"

EVENT = {
    "time":    "19:00",
    "weather": "맑음",
    "date":    "2026-12-25",
    "country": "대한민국",
    "city":    "서울시",
    "place":   "무궁화 아파트",
}

ROWS = [
    ("00_data_only_no_lora",   None),
    ("01_data_only_worstimever",  "worstimever_xl.safetensors"),
    ("02_data_only_mspaint",      "sdxl_mspaint_portraits.safetensors"),
]

SEED = 42


def main() -> None:
    ep = build_event_prompt(EVENT)
    # Reuse the breakdown to get JUST the data phrases — drop the
    # LORA_TRIGGER and STYLE_TAGS that build_event_prompt would have
    # otherwise prepended.
    moments = [phrase for _src, phrase in ep.breakdown]
    data_only_prompt = ", ".join(moments)
    print(f"[data] event: {EVENT}")
    print(f"[data] data-only prompt: {data_only_prompt}\n")

    print("[data] loading SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    md = [
        "# Data-only baseline — just translated event, no styling",
        "",
        "User wanted to see: 'what does SDXL produce if I just give it",
        "my translated event data, no STYLE_TAGS, no LoRA trigger phrase,",
        "no NEGATIVE prompt?'",
        "",
        "Input event: 19:00 / 맑음 / 2026-12-25 / 대한민국 서울시 무궁화 아파트  ",
        f"Data-only prompt: `{data_only_prompt}`",
        "",
        "| LoRA fused | result |",
        "| --- | --- |",
    ]

    for name, fname in ROWS:
        try: pipe.unfuse_lora()
        except Exception: pass
        try: pipe.unload_lora_weights()
        except Exception: pass

        if fname:
            pipe.load_lora_weights(str(LORA_DIR), weight_name=fname, adapter_name=name)
            pipe.fuse_lora(lora_scale=0.9)

        print(f"\n[data] === {name} ===")
        t = time.time()
        out = pipe(
            prompt=data_only_prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        img = out.images[0]
        img_path = SAMPLES / f"data_only_{name}.png"
        img.save(img_path)
        print(f"[data]   {time.time() - t:.1f}s -> {img_path.name}")

        lora_disp = fname if fname else "(no LoRA)"
        md.append(f"| `{lora_disp}` | ![]({img_path.name}) |")

    (SAMPLES / "data_only_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[data] index -> samples/data_only_grid.md")


if __name__ == "__main__":
    main()
