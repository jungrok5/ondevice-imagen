"""Phase 2 Step 6-A — UNet only, full fp16 (input/output too).

The Step 3 attempt (quantize_onnx_step3.py) tried keep_io_types=True
so the existing OnnxStableDiffusionPipeline would still work with
fp32 IO. That path lost: every variant hit "Type Error: Type
parameter (T) of Optype (...) bound to different types
(tensor(float16) and tensor(float))" because keep_io_types=True
preserves the fp32 attention-mask / timestep / etc. inputs and
those get fed into ops alongside fp16 weights.

Step 6-A is different: we *control the ORT session manually* in
SdInferencePipeline.kt now (since Step 5c). No need for
OnnxStableDiffusionPipeline. We can pass fp16 IO tensors directly,
which means casting the IO too is fine (and necessary).

Output: models/onnx_fp16_unet/sd15_drawing_nty_scale0.8/unet/
        model.onnx (graph) + model.onnx_data (fp16 weights, ~1.7 GB)

Other sub-models (text_encoder, vae_*) stay fp32 — the type-binding
problem still bites them under keep_io_types=False because their
input shapes are part of the public API the diffusers wrappers
assume. Phone-side runUnetCfg will cast text_encoder hidden_states
and the latent down to fp16 before feeding UNet, then cast UNet's
fp16 noise_pred back to fp32 for the scheduler step.

UNet alone is 77% of the bundle, so cutting it from 3.3 GB to
~1.7 GB is most of the memory headroom we need to dodge LMK on
8 GB phones.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models" / "onnx"             / "sd15_drawing_nty_scale0.8"
DST_DIR = ROOT / "models" / "onnx_fp16_unet"   / "sd15_drawing_nty_scale0.8"


def cast_unet_full_fp16() -> None:
    src = SRC_DIR / "unet" / "model.onnx"
    dst = DST_DIR / "unet" / "model.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fp16-unet] loading {src}")
    t = time.time()
    model = onnx.load(str(src), load_external_data=True)
    print(f"[fp16-unet]   loaded in {time.time() - t:.1f}s")

    print(f"[fp16-unet] casting to fp16 (keep_io_types=False)")
    t = time.time()
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,        # full fp16 graph + IO
        disable_shape_infer=True,
    )
    print(f"[fp16-unet]   cast in {time.time() - t:.1f}s")

    print(f"[fp16-unet] saving with external data to {dst}")
    t = time.time()
    onnx.save(
        fp16_model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        convert_attribute=False,
    )
    print(f"[fp16-unet]   saved in {time.time() - t:.1f}s")


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"missing src ONNX dir — run Step 2 first: {SRC_DIR}")

    cast_unet_full_fp16()

    # Copy the rest as-is so this directory is a drop-in replacement
    # for the fp32 bundle — same model_index.json, scheduler/, etc.
    for sub in ("text_encoder", "vae_encoder", "vae_decoder",
                "scheduler", "tokenizer"):
        s, d = SRC_DIR / sub, DST_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
    shutil.copy(SRC_DIR / "model_index.json", DST_DIR / "model_index.json")

    total = sum(f.stat().st_size for f in DST_DIR.rglob("*") if f.is_file())
    src_total = sum(f.stat().st_size for f in SRC_DIR.rglob("*") if f.is_file())
    print(f"\n[fp16-unet] DONE")
    print(f"  fp32 src:  {src_total / 1_000_000_000:.2f} GB")
    print(f"  fp16-unet: {total     / 1_000_000_000:.2f} GB "
          f"({100 * total / src_total:.0f}% of fp32)")
    for sub in sorted(DST_DIR.iterdir()):
        if sub.is_dir():
            sub_size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            print(f"  {sub.name:20s} {sub_size / 1_000_000:8.1f} MB")


if __name__ == "__main__":
    main()
