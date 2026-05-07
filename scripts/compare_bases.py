"""Compare two commercial-friendly SDXL bases × two style LoRAs.

Bases:
  - SDXL 1.0 base   30 step / guidance 7   (Open RAIL++ M, commercial OK)
  - SDXL Lightning  4 step / guidance 0    (Open RAIL++ M, commercial OK)
                    distributed as a LoRA over SDXL 1.0 base

Style LoRAs (same as the shootout):
  - worstimever         trigger 'DD-wte artstyle, worst-im-ever cartoon doodle'
  - mspaint_portraits   trigger 'MSPaint drawing'

That makes 4 cells. SDXL Lightning is loaded as a LoRA *alongside* the
style LoRA via PEFT adapter combine.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler

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
    "city":    "수원시",
    "place":   "광교포레스트 아파트",
}

STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, naive crude drawing"
)
NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)

STYLE_LORAS = [
    ("worstimever",       "worstimever_xl.safetensors",       "DD-wte artstyle, worst-im-ever cartoon doodle"),
    ("mspaint_portraits", "sdxl_mspaint_portraits.safetensors", "MSPaint drawing"),
]


def event_phrases(event: dict) -> str:
    ep = build_event_prompt(event)
    moments = [phrase for _src, phrase in ep.breakdown]
    return ", ".join(moments)


def load_base() -> AutoPipelineForText2Image:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to("cpu")


def reset_loras(pipe) -> None:
    try:
        pipe.unfuse_lora()
    except Exception:
        pass
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass


def generate(
    pipe,
    *,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int = 42,
):
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=512,
        height=512,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


def main() -> None:
    moments = event_phrases(EVENT)
    print(f"[bases] data phrases: {moments}\n")

    print(f"[bases] loading SDXL 1.0 base...")
    pipe = load_base()
    print(f"[bases] base loaded.")

    md = [
        "# Commercial-friendly base × style LoRA shootout",
        "",
        "Same SDXL 1.0 base (Open RAIL++ M, commercial OK). Two speed",
        "regimes — full 30-step base inference vs SDXL Lightning",
        "(loaded as a LoRA over the same base, 4 step). Two style LoRAs",
        "stacked on top of each.",
        "",
        f"Input event: 19:00 / 맑음 / 2026-12-25 / 광교포레스트 아파트  ",
        f"Data phrases: `{moments}`",
        "",
        "| base regime | style LoRA | result |",
        "| --- | --- | --- |",
    ]

    # Loop: 4 cells = 2 regimes × 2 style LoRAs
    regimes = [
        ("base_30step",     None,                                    30, 7.0,  None),
        ("lightning_4step", "sdxl_lightning_4step_lora.safetensors",  4, 0.0,  "lightning"),
    ]

    for regime_name, light_lora, steps, guidance, light_adapter in regimes:
        # Switch scheduler for Lightning (uses Euler with trailing timesteps)
        if regime_name == "lightning_4step":
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

        for style_name, style_file, style_trigger in STYLE_LORAS:
            print(f"\n[bases] === {regime_name} + {style_name} ===")
            reset_loras(pipe)

            adapter_names = []
            adapter_weights = []

            if light_lora:
                pipe.load_lora_weights(
                    str(LORA_DIR), weight_name=light_lora,
                    adapter_name="lightning",
                )
                adapter_names.append("lightning")
                adapter_weights.append(1.0)

            pipe.load_lora_weights(
                str(LORA_DIR), weight_name=style_file,
                adapter_name=style_name,
            )
            adapter_names.append(style_name)
            adapter_weights.append(0.9)

            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            pipe.fuse_lora()

            prompt = f"{style_trigger}, {STYLE_TAGS}, {moments}"

            t = time.time()
            img = generate(
                pipe, prompt=prompt, steps=steps, guidance=guidance, seed=42
            )
            dt = time.time() - t
            out_path = SAMPLES / f"base_compare_{regime_name}_{style_name}.png"
            img.save(out_path)
            print(f"[bases]   {dt:.1f}s -> {out_path.name}")

            md.append(
                f"| `{regime_name}` ({steps} step) "
                f"| `{style_name}` "
                f"| ![]({out_path.name}) |"
            )

    (SAMPLES / "base_compare_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[bases] index -> samples/base_compare_grid.md")


if __name__ == "__main__":
    main()
