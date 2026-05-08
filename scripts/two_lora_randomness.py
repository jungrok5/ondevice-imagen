"""Same event × 2 style LoRAs × N calls = 2*N different images.

Demonstrates that the prompt randomness in src/event_prompt.py works
under both supported style LoRAs (worstimever, mspaint_portraits),
not just one. Per the testing rule, each LoRA is loaded with its
card-specified trigger phrase auto-prepended via LORA_TRIGGERS.

Diffusion seed varies per cell so the only constant is the EVENT
dict; phrase randomness + diffusion randomness combine.
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
LORAS = [
    ("worstimever",       "worstimever_xl.safetensors"),
    ("mspaint_portraits", "sdxl_mspaint_portraits.safetensors"),
]
N_PER_LORA = 4


def main() -> None:
    print("[rand2] loading SDXL-Turbo (cached)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    md_lines = [
        f"# Two-LoRA randomness check — {N_PER_LORA} calls each",
        "",
        f"Same `EVENT` dict, both style LoRAs (`worstimever`,",
        "`mspaint_portraits`), {N} calls per LoRA. Each call draws".format(
            N=N_PER_LORA
        ),
        "fresh phrases from the pools in `src/event_prompt.py`. The",
        "diffusion seed also varies per cell so prompt + denoise",
        "randomness combine.",
        "",
        f"| call # | {LORAS[0][0]} | {LORAS[1][0]} |",
        "| --- | --- | --- |",
    ]

    # Generate per-LoRA, then re-arrange into rows for the grid
    cell_paths: dict[tuple[str, int], str] = {}

    for style_name, lora_file in LORAS:
        print(f"\n[rand2] === fusing {style_name} ===")
        try: pipe.unfuse_lora()
        except Exception: pass
        try: pipe.unload_lora_weights()
        except Exception: pass

        pipe.load_lora_weights(
            str(LORA_DIR), weight_name=lora_file, adapter_name=style_name
        )
        pipe.fuse_lora(lora_scale=0.9)

        for i in range(N_PER_LORA):
            built = build_event_prompt(EVENT, visual_style=style_name)
            gen_seed = 2000 + i  # constant across LoRAs so we compare apples-to-apples
            print(f"\n[rand2] --- {style_name} sample {i} (gen_seed={gen_seed}) ---")
            print(f"[rand2] prompt: {built.positive[:160]}...")

            t = time.time()
            img = pipe(
                prompt=built.positive,
                negative_prompt=built.negative,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=512,
                height=512,
                generator=torch.Generator(device="cpu").manual_seed(gen_seed),
            ).images[0]
            out_path = SAMPLES / f"rand2_{style_name}_{i}.png"
            img.save(out_path)
            print(f"[rand2]   {time.time() - t:.1f}s -> {out_path.name}")
            cell_paths[(style_name, i)] = out_path.name

    # Build the row table — one row per call index, two columns (one per LoRA)
    for i in range(N_PER_LORA):
        row = [f"| {i} |"]
        for style_name, _lora_file in LORAS:
            row.append(f" ![]({cell_paths[(style_name, i)]}) |")
        md_lines.append("".join(row))

    (SAMPLES / "rand2_grid.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\n[rand2] index -> samples/rand2_grid.md")


if __name__ == "__main__":
    main()
