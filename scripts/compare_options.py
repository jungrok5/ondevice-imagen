"""Try all 4 fixes A/B/C/D for the Lightning + style-LoRA quality gap.

Single style LoRA: worstimever (the user's preferred trend register).
Single base: SDXL 1.0 base (commercial OK).
What changes per option is HOW Lightning is applied (or replaced):

  A — sdxl_lightning_4step_unet.safetensors  (full UNet replacement,
      no LoRA conflict with worstimever)
  B — sdxl_lightning_8step_lora.safetensors  (8-step gives more denoise
      headroom for the style LoRA to express)
  C — sdxl_lightning_4step_lora at scale 1.0 + worstimever at scale 1.2
      (force the style LoRA stronger to overpower the Lightning LoRA's
      grip on the same UNet weights)
  D — hyper_sdxl_4step_lora.safetensors      (different distillation;
      Hyper-SD's adversarial distillation may stack better with style LoRAs)

All 4 use the same Christmas-event prompt + seed. UNet variant (option A)
is loaded only if the file exists when the script reaches it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
    TCDScheduler,
)
from safetensors.torch import load_file

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

STYLE_TAGS = (
    "ugly MS Paint doodle, white paper, black ink only, pixelated low-res, "
    "child scribble, naive crude drawing"
)
NEGATIVE = (
    "photorealistic, sharp focus, polished, professional, hd, "
    "anti-aliased, smooth gradient, oil painting"
)
WORSTIMEVER_TRIGGER = "DD-wte artstyle, worst-im-ever cartoon doodle"


def event_phrases(event: dict) -> str:
    ep = build_event_prompt(event)
    return ", ".join(phrase for _src, phrase in ep.breakdown)


def reset_loras(pipe) -> None:
    try: pipe.unfuse_lora()
    except Exception: pass
    try: pipe.unload_lora_weights()
    except Exception: pass


def fresh_pipe() -> AutoPipelineForText2Image:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to("cpu")


def gen(pipe, *, prompt: str, steps: int, guidance: float, seed: int = 42):
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
    prompt = f"{WORSTIMEVER_TRIGGER}, {STYLE_TAGS}, {moments}"
    print(f"[opts] prompt: {prompt[:140]}...\n")

    md_rows: list[str] = []

    # ---------- B: Lightning 8-step LoRA ----------
    print("\n[opts] === B: Lightning 8-step LoRA ===")
    pipe = fresh_pipe()
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.load_lora_weights(str(LORA_DIR), weight_name="sdxl_lightning_8step_lora.safetensors", adapter_name="lightning")
    pipe.load_lora_weights(str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="worst")
    pipe.set_adapters(["lightning", "worst"], adapter_weights=[1.0, 0.9])
    pipe.fuse_lora()
    t = time.time()
    img = gen(pipe, prompt=prompt, steps=8, guidance=0.0)
    out = SAMPLES / "opt_B_lightning_8step.png"
    img.save(out); print(f"[opts]   {time.time()-t:.1f}s -> {out.name}")
    md_rows.append(("B — Lightning 8-step + worst@0.9", out.name))

    # ---------- C: Lightning 4-step LoRA + worst at scale 1.2 ----------
    print("\n[opts] === C: Lightning 4-step + worstimever @ 1.2 ===")
    pipe = fresh_pipe()
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.load_lora_weights(str(LORA_DIR), weight_name="sdxl_lightning_4step_lora.safetensors", adapter_name="lightning")
    pipe.load_lora_weights(str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="worst")
    pipe.set_adapters(["lightning", "worst"], adapter_weights=[1.0, 1.2])
    pipe.fuse_lora()
    t = time.time()
    img = gen(pipe, prompt=prompt, steps=4, guidance=0.0)
    out = SAMPLES / "opt_C_worst_scale12.png"
    img.save(out); print(f"[opts]   {time.time()-t:.1f}s -> {out.name}")
    md_rows.append(("C — Lightning 4-step + worst@1.2", out.name))

    # ---------- D: Hyper-SD 4-step LoRA ----------
    print("\n[opts] === D: Hyper-SD 4-step LoRA + worstimever ===")
    pipe = fresh_pipe()
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA_DIR), weight_name="hyper_sdxl_4step_lora.safetensors", adapter_name="hyper")
    pipe.load_lora_weights(str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="worst")
    pipe.set_adapters(["hyper", "worst"], adapter_weights=[1.0, 0.9])
    pipe.fuse_lora()
    t = time.time()
    img = gen(pipe, prompt=prompt, steps=4, guidance=0.0)
    out = SAMPLES / "opt_D_hyper.png"
    img.save(out); print(f"[opts]   {time.time()-t:.1f}s -> {out.name}")
    md_rows.append(("D — Hyper-SD 4-step + worst@0.9", out.name))

    # ---------- A: Lightning UNet variant replacement ----------
    unet_file = LORA_DIR / "sdxl_lightning_4step_unet.safetensors"
    if unet_file.exists() and unet_file.stat().st_size > 1_000_000_000:
        print("\n[opts] === A: Lightning UNet variant (full UNet replacement) ===")
        pipe = fresh_pipe()
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        # Replace UNet with the Lightning-distilled one
        pipe.unet = UNet2DConditionModel.from_config(pipe.unet.config).to("cpu", torch.float32)
        pipe.unet.load_state_dict(load_file(str(unet_file), device="cpu"))
        # Now only the style LoRA, no LoRA conflict
        pipe.load_lora_weights(str(LORA_DIR), weight_name="worstimever_xl.safetensors", adapter_name="worst")
        pipe.set_adapters(["worst"], adapter_weights=[0.9])
        pipe.fuse_lora()
        t = time.time()
        img = gen(pipe, prompt=prompt, steps=4, guidance=0.0)
        out = SAMPLES / "opt_A_lightning_unet.png"
        img.save(out); print(f"[opts]   {time.time()-t:.1f}s -> {out.name}")
        md_rows.append(("A — Lightning UNet variant + worst@0.9 (no LoRA conflict)", out.name))
    else:
        print("\n[opts] A skipped (Lightning 4-step UNet file not yet downloaded)")

    md = [
        "# Lightning + style-LoRA quality fixes — A/B/C/D shootout",
        "",
        f"Same Christmas event, same prompt, same seed (42), same style LoRA",
        f"(worstimever). Only the *speed-distillation method* differs per row.",
        "",
        f"Data phrases injected: `{moments}`",
        "",
        "| option | result |",
        "| --- | --- |",
    ]
    for label, fname in md_rows:
        md.append(f"| {label} | ![]({fname}) |")
    (SAMPLES / "options_compare_grid.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"\n[opts] index -> samples/options_compare_grid.md")


if __name__ == "__main__":
    main()
