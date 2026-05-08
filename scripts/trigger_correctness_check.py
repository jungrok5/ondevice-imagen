"""Trigger correctness — old (our previous guess) vs new (Civitai-official).

Civitai pages are the only reliable trigger source. We had been using
guess-strings ("DD-wte artstyle..." for worstimever, "MSPaint drawing of"
for mspaint_portraits). This script renders the same exact event with
the same phrase pick (prompt_seed) and same diffusion seed for each
LoRA, swapping ONLY the trigger phrase, so any visible difference is
solely the trigger's effect on cross-attention conditioning.

  worstimever:       OLD = "DD-wte artstyle, worst-im-ever cartoon doodle"
                     NEW = "WTE artstyle"
  mspaint_portraits: OLD = "MSPaint drawing of"
                     NEW = "MSPaint portrait"
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
PROMPT_SEED = 100      # fixes the phrase pool draw so every cell has the SAME data prompt
DIFFUSION_SEED = 42    # fixes denoise noise

CASES = [
    # (lora_label, lora_file, trigger, variant_tag)
    ("worstimever",       "worstimever_xl.safetensors",
     "DD-wte artstyle, worst-im-ever cartoon doodle", "OLD"),
    ("worstimever",       "worstimever_xl.safetensors",
     "WTE artstyle", "NEW"),
    ("mspaint_portraits", "sdxl_mspaint_portraits.safetensors",
     "MSPaint drawing of", "OLD"),
    ("mspaint_portraits", "sdxl_mspaint_portraits.safetensors",
     "MSPaint portrait", "NEW"),
]


def main() -> None:
    print("[trig] loading SDXL-Turbo (cached)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    cell_paths: dict[tuple[str, str], str] = {}

    current_lora_label: str | None = None

    for lora_label, lora_file, trigger, variant_tag in CASES:
        # Re-fuse only when the LoRA itself changes (saves a fuse cycle
        # for the same LoRA across two trigger variants).
        if lora_label != current_lora_label:
            print(f"\n[trig] === fusing {lora_label} ===")
            try: pipe.unfuse_lora()
            except Exception: pass
            try: pipe.unload_lora_weights()
            except Exception: pass
            pipe.load_lora_weights(
                str(LORA_DIR), weight_name=lora_file, adapter_name=lora_label
            )
            pipe.fuse_lora(lora_scale=0.9)
            current_lora_label = lora_label

        # Build the data block with NO trigger (visual_style="none")
        # using a fixed prompt_seed so every cell has the same data text.
        # Then we prepend the variant trigger ourselves.
        ep = build_event_prompt(
            EVENT, visual_style="none", include_style_tags=True,
            prompt_seed=PROMPT_SEED,
        )
        positive = f"{trigger}, {ep.positive}"

        print(f"\n[trig] --- {lora_label} {variant_tag} ---")
        print(f"[trig] prompt: {positive[:160]}...")

        t = time.time()
        img = pipe(
            prompt=positive,
            negative_prompt=ep.negative,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(DIFFUSION_SEED),
        ).images[0]
        out_path = SAMPLES / f"trig_{lora_label}_{variant_tag}.png"
        img.save(out_path)
        print(f"[trig]   {time.time() - t:.1f}s -> {out_path.name}")
        cell_paths[(lora_label, variant_tag)] = out_path.name

    md_lines = [
        "# Trigger correctness — old (guess) vs new (Civitai-official)",
        "",
        "Same `EVENT`, same `prompt_seed=100`, same `DIFFUSION_SEED=42`.",
        "Only the trigger phrase changes between cells in a row.",
        "",
        "| LoRA | OLD trigger | NEW trigger |",
        "| --- | --- | --- |",
        f"| **worstimever** | "
        f"![]({cell_paths[('worstimever','OLD')]})<br>"
        f"`DD-wte artstyle, worst-im-ever cartoon doodle` | "
        f"![]({cell_paths[('worstimever','NEW')]})<br>"
        f"`WTE artstyle` |",
        f"| **mspaint_portraits** | "
        f"![]({cell_paths[('mspaint_portraits','OLD')]})<br>"
        f"`MSPaint drawing of` | "
        f"![]({cell_paths[('mspaint_portraits','NEW')]})<br>"
        f"`MSPaint portrait` |",
    ]
    (SAMPLES / "trig_grid.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\n[trig] index -> samples/trig_grid.md")


if __name__ == "__main__":
    main()
