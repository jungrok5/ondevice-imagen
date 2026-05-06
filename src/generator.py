"""Stable Diffusion wrappers — txt2img and img2img.

Default model: stabilityai/sd-turbo
- ~1.4 GB on disk (fp16)
- Generates a 512x512 image in 1-4 steps, even on CPU
- Works on Windows + AMD via plain PyTorch CPU; opt-in DirectML for speedup

This module is lazy: the pipeline is only loaded the first time `generate*` is
called. Both pipelines share the underlying weights via `components`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


DEFAULT_MODEL = os.environ.get("LAR_MODEL_ID", "stabilityai/sd-turbo")
DEFAULT_STEPS = int(os.environ.get("LAR_STEPS", "2"))
DEFAULT_GUIDANCE = float(os.environ.get("LAR_GUIDANCE", "0.0"))  # SD-Turbo wants 0
DEFAULT_STRENGTH = float(os.environ.get("LAR_STRENGTH", "0.85"))


def _resolve_device() -> tuple[str, torch.dtype]:
    """Pick the best available device; explain the choice via env LAR_DEVICE."""
    forced = os.environ.get("LAR_DEVICE", "").lower()
    if forced == "cpu":
        return "cpu", torch.float32
    if forced == "directml":
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), torch.float16
        except ImportError as e:
            raise RuntimeError(
                "LAR_DEVICE=directml requested but torch-directml is not installed. "
                "Install with: py -m pip install torch-directml"
            ) from e
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # Auto: try DirectML if installed (helps on AMD/Intel GPU on Windows)
    try:
        import torch_directml  # type: ignore

        return torch_directml.device(), torch.float16
    except ImportError:
        return "cpu", torch.float32


@dataclass
class _PipelineBundle:
    txt2img: object
    img2img: object
    device: object
    dtype: torch.dtype


_BUNDLE: Optional[_PipelineBundle] = None


def _load() -> _PipelineBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    from diffusers import (
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
    )

    device, dtype = _resolve_device()
    print(f"[generator] loading {DEFAULT_MODEL} on {device} ({dtype})")

    txt2img = AutoPipelineForText2Image.from_pretrained(
        DEFAULT_MODEL, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    txt2img = txt2img.to(device)

    # Reuse weights for img2img — no double download / no double VRAM
    img2img = AutoPipelineForImage2Image.from_pipe(txt2img).to(device)

    _BUNDLE = _PipelineBundle(txt2img=txt2img, img2img=img2img, device=device, dtype=dtype)
    return _BUNDLE


def generate_txt2img(
    prompt: str,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> Image.Image:
    bundle = _load()
    generator = None
    if seed is not None:
        # Use CPU generator for reproducibility regardless of device
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    out = bundle.txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    return out.images[0]


def generate_img2img(
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    strength: float = DEFAULT_STRENGTH,
    seed: Optional[int] = None,
) -> Image.Image:
    bundle = _load()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    init = init_image.convert("RGB").resize((512, 512))
    out = bundle.img2img(
        prompt=prompt,
        image=init,
        negative_prompt=negative_prompt or None,
        num_inference_steps=max(steps, 2),  # img2img needs >=2 effective steps
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    )
    return out.images[0]
