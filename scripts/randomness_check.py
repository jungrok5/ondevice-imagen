"""Same event × 4 generations = 4 different images.

Demonstrates the prompt randomness added to src/event_prompt.py.
Each call draws different phrase choices from the pools, so the
final image varies even though the input data is identical.

We also vary the diffusion seed per cell so the only constant is
the *event dict*. To isolate just-prompt-randomness, set SAME_DIFFUSION_SEED=True.
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
N_SAMPLES = 4
SAME_DIFFUSION_SEED = False  # if True, only prompt randomness varies


def main() -> None:
    print("[rand] loading SDXL-Turbo (cached)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    pipe.load_lora_weights(
        str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="w"
    )
    pipe.fuse_lora(lora_scale=0.9)

    md_lines = [
        f"# Randomness check — same event × {N_SAMPLES} calls",
        "",
        "Same exact `EVENT` dict, same LoRA, but each call to",
        "`build_event_prompt()` draws different phrases from the pools",
        "in `src/event_prompt.py`. Result: 4 different images from",
        "identical input.",
        "",
        f"Diffusion seed varies per cell: `SAME_DIFFUSION_SEED={SAME_DIFFUSION_SEED}`",
        "",
        "| # | prompt (excerpt) | result |",
        "| --- | --- | --- |",
    ]

    for i in range(N_SAMPLES):
        built = build_event_prompt(EVENT, visual_style="worstimever")
        # pick a different diffusion seed each cell (unless flag)
        gen_seed = 42 if SAME_DIFFUSION_SEED else 1000 + i
        print(f"\n[rand] === sample {i} (gen_seed={gen_seed}) ===")
        print(f"[rand] prompt: {built.positive[:160]}...")

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
        out_path = SAMPLES / f"rand_{i}.png"
        img.save(out_path)
        print(f"[rand]   {time.time() - t:.1f}s -> {out_path.name}")

        # Excerpt the user-data portion (skip the trigger prefix so
        # you can SEE what differs cell-to-cell)
        TRIGGER_END = "worst-im-ever cartoon doodle, "
        body = built.positive.split(TRIGGER_END, 1)[-1]
        # And cut before the generic style tags tail
        body = body.split(", ugly MS Paint doodle", 1)[0]
        md_lines.append(
            f"| {i} | {body[:140]}{'...' if len(body) > 140 else ''} | ![](rand_{i}.png) |"
        )

    (SAMPLES / "rand_grid.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(f"\n[rand] index -> samples/rand_grid.md")


if __name__ == "__main__":
    main()
