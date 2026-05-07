"""Run the same single-event input through every available SDXL LoRA
and produce a side-by-side comparison board.

Same event (user's literal Christmas example), same SDXL-Turbo seed,
only the loaded LoRA changes per row.
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


# (display name, file, optional trigger phrase to PREPEND, lora scale)
LORA_TABLE = [
    ("worstimever",        "worstimever_xl.safetensors",      "DD-wte artstyle, worst-im-ever cartoon doodle", 0.9),
    ("mspaint_portraits",  "sdxl_mspaint_portraits.safetensors", "MSPaint drawing",                            0.9),
    ("lah_cute_social",    "lah_cute_social.safetensors",     "cute doodle",                                   0.9),
    ("pixel_art_xl",       "pixel_art_xl.safetensors",         None,                                            0.9),
]


EVENT = {
    "time":    "19:00",
    "weather": "맑음",
    "date":    "2026-12-25",
    "country": "대한민국",
    "city":    "서울시",
    "place":   "무궁화 아파트",
}


# Style block from event_prompt.py reused but allow per-LoRA trigger to
# go FIRST so it dominates CLIP's 77-token budget.
STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, naive crude drawing"
)
NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)


def event_phrase(event: dict) -> str:
    """Return only the data-driven phrases (no style tags), since each
    LoRA brings its own style. event_prompt's positive includes the
    DD-wte trigger we want to override per-LoRA, so reuse just the
    breakdown phrases."""
    ep = build_event_prompt(event)
    # Drop the DD-wte trigger and STYLE_TAGS from the positive — keep
    # only the moments. Easiest: reconstruct from breakdown.
    moments = [phrase for _src, phrase in ep.breakdown]
    return ", ".join(moments)


def main() -> None:
    moments_block = event_phrase(EVENT)
    print(f"[compare] data phrases: {moments_block}\n")

    print(f"[compare] loading SDXL-Turbo (one base, swap LoRAs)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    md = [
        "# LoRA shootout — same event, 4 LoRAs",
        "",
        "Input event: 19:00 / 맑음 / 2026-12-25 / 대한민국 수원시 광교포레스트 아파트",
        "",
        "Data phrases injected: `" + moments_block + "`",
        "",
        "Same SDXL-Turbo base, same prompt structure, same seed (42). The",
        "only thing changing per row is the loaded LoRA + its trigger.",
        "",
        "| LoRA | trigger | result |",
        "| --- | --- | --- |",
    ]

    for name, fname, trigger, scale in LORA_TABLE:
        lora_path = LORA_DIR / fname
        if not lora_path.exists():
            print(f"[compare] missing {fname}, skipping")
            continue

        print(f"\n[compare] === {name} (scale={scale}) ===")
        # Hard reset adapters before each load
        try:
            pipe.unfuse_lora()
        except Exception:
            pass
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        pipe.load_lora_weights(
            str(LORA_DIR), weight_name=fname, adapter_name=name
        )
        pipe.fuse_lora(lora_scale=scale)

        if trigger:
            prompt = f"{trigger}, {STYLE_TAGS}, {moments_block}"
        else:
            prompt = f"{STYLE_TAGS}, {moments_block}"
        print(f"[compare] prompt: {prompt[:140]}...")

        t = time.time()
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        img = out.images[0]
        img_path = SAMPLES / f"lora_compare_{name}.png"
        img.save(img_path)
        print(f"[compare]   {time.time() - t:.1f}s -> {img_path.name}")

        trigger_disp = trigger if trigger else "(no trigger)"
        md.append(f"| `{name}` | `{trigger_disp}` | ![]({img_path.name}) |")

    (SAMPLES / "lora_compare_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[compare] index -> samples/lora_compare_grid.md")


if __name__ == "__main__":
    main()
