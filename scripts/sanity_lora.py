"""Sanity test — does fuse_lora() with trigger actually change output?

Two cells, same exact data prompt:
  A: no LoRA at all
  B: worstimever fused @ 0.9 + trigger phrase prepended

If A vs B differ noticeably -> LoRA is working; the previous test's
'all 3 look the same' just meant "fused but trigger absent ≈ no effect".
If A vs B look identical -> something is wrong with fuse/unload state.

A third cell C: same as B but unfuse + reload nothing in between, to
confirm unfuse + unload is real.
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
SEED = 42

LORA_TRIGGER = "DD-wte artstyle, worst-im-ever cartoon doodle"


def main() -> None:
    ep = build_event_prompt(EVENT)
    moments = ", ".join(phrase for _src, phrase in ep.breakdown)
    base_prompt = moments
    triggered_prompt = f"{LORA_TRIGGER}, {moments}"
    print(f"[sanity] base prompt: {base_prompt}")
    print(f"[sanity] triggered : {triggered_prompt}\n")

    print("[sanity] loading SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    def gen(prompt: str, label: str) -> None:
        print(f"\n[sanity] === {label} ===")
        print(f"[sanity] prompt: {prompt}")
        t = time.time()
        img = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        ).images[0]
        path = SAMPLES / f"sanity_{label}.png"
        img.save(path)
        print(f"[sanity]   {time.time() - t:.1f}s -> {path.name}")

    # A: NO LoRA, base prompt
    gen(base_prompt, "A_no_lora_base_prompt")

    # B: LoRA fused + trigger in prompt
    pipe.load_lora_weights(
        str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="w"
    )
    pipe.fuse_lora(lora_scale=0.9)
    gen(triggered_prompt, "B_worst_fused_with_trigger")

    # C: same LoRA still fused + base prompt (NO trigger)
    gen(base_prompt, "C_worst_fused_no_trigger")

    # D: unfuse + unload, then base prompt (should match A if unload works)
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    gen(base_prompt, "D_after_unload_base_prompt")

    md = [
        "# LoRA fuse/unload sanity check",
        "",
        "Same SDXL-Turbo, same data prompt, same seed. Only the LoRA",
        "state and trigger phrase change row-to-row.",
        "",
        "| label | LoRA state | prompt | result |",
        "| --- | --- | --- | --- |",
        f"| A | none | data only | ![](sanity_A_no_lora_base_prompt.png) |",
        f"| B | worstimever fused @ 0.9 | trigger + data | ![](sanity_B_worst_fused_with_trigger.png) |",
        f"| C | worstimever still fused | data only (NO trigger) | ![](sanity_C_worst_fused_no_trigger.png) |",
        f"| D | after unfuse + unload | data only | ![](sanity_D_after_unload_base_prompt.png) |",
        "",
        "Expected:",
        "  - A and D should match (LoRA fully unloaded by D)",
        "  - B should differ noticeably from A/D (trigger activates style)",
        "  - C tells us how much LoRA does WITHOUT trigger — small or large?",
    ]
    (SAMPLES / "sanity_grid.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[sanity] index -> samples/sanity_grid.md")


if __name__ == "__main__":
    main()
