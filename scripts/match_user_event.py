"""Run the user's exact example through src/event_prompt.py + SDXL-Turbo
+ worstimever LoRA, and dump the prompt + breakdown + image.

Example event:
  19:00 / 맑음 / 2026-12-25 / 대한민국 수원시 광교포레스트 아파트
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
LORA_PATH = ROOT / "models" / "lora" / "worstimever_xl.safetensors"


EVENT = {
    "time":    "19:00",
    "weather": "맑음",
    "date":    "2026-12-25",
    "country": "대한민국",
    "city":    "서울시",
    "place":   "무궁화 아파트",
}


def main() -> None:
    ep = build_event_prompt(EVENT)
    print("[event] input:")
    for k, v in EVENT.items():
        print(f"    {k}: {v}")
    print("\n[event] field → phrase breakdown:")
    for src, phrase in ep.breakdown:
        print(f"    {src}  →  {phrase}")
    print(f"\n[event] full positive prompt:\n    {ep.positive}\n")
    print(f"[event] negative prompt:\n    {ep.negative}\n")

    print("[event] loading SDXL-Turbo + worstimever LoRA...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    pipe.load_lora_weights(
        str(LORA_PATH.parent), weight_name=LORA_PATH.name
    )
    pipe.fuse_lora(lora_scale=0.9)
    print("[event] generating...")
    t = time.time()
    out = pipe(
        prompt=ep.positive,
        negative_prompt=ep.negative,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512,
    )
    img = out.images[0]
    img_path = SAMPLES / "event_xmas_19h.png"
    img.save(img_path)
    print(f"[event] {time.time() - t:.1f}s -> {img_path}")

    md = [
        "# Single-event prompt: 2026-12-25 19:00 광교포레스트",
        "",
        "## Input event",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for k, v in EVENT.items():
        md.append(f"| {k} | {v} |")

    md += [
        "",
        "## Field → phrase mapping (from src/event_prompt.py)",
        "",
        "| source field | phrase injected |",
        "| --- | --- |",
    ]
    for src, phrase in ep.breakdown:
        md.append(f"| `{src}` | `{phrase}` |")

    md += [
        "",
        "## Full positive prompt (sent to SDXL-Turbo + worstimever LoRA)",
        "",
        "```",
        ep.positive,
        "```",
        "",
        "## Negative prompt",
        "",
        "```",
        ep.negative,
        "```",
        "",
        "## Result",
        "",
        f"![]({img_path.name})",
    ]
    (SAMPLES / "event_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[event] index -> samples/event_grid.md")


if __name__ == "__main__":
    main()
