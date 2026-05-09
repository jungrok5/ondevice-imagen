"""Phase 2 Step 3 — fp16 cast of the fp32 ONNX export.

Target: knock the 4.1 GB fp32 ONNX export down to ~2 GB so it fits
in a phone download budget. fp16 has ~zero quality cost on SD 1.5
in our testing tier — INT8 dynamic would go further (~1 GB) but
typically degrades the VAE decoder badly (noise / banding) so we
defer it. NNAPI / QNN both accept fp16 directly on the Note 10+
Mali / Hexagon hardware.

STATUS: blocked on this stack. Committed as a documented attempt
+ reproducer for the Kotlin / mobile retry in Step 5.

What works conceptually: cast Step 2's fp32 weights to fp16 to get
4.27 GB → ~2 GB, fitting a phone download budget. fp16 has ~zero
quality cost on SD 1.5 in our testing tier; INT8 dynamic would go
further (~1 GB) but historically degrades the VAE decoder badly.
NNAPI / QNN both accept fp16 directly on Note 10+ Mali / Hexagon.

What blocks it on this PC stack (diffusers 0.30.3 + onnxruntime 1.19.2
+ onnxconverter-common 1.16.0 + Python 3.9):

  - All sub-models cast: VAE self-attn `Div` op binds fp16 weights
    against the fp32 IO tensor preserved by `keep_io_types=True`.
  - VAE skipped, UNet+text_encoder cast: CLIP self_attn `Add` binds
    fp16 weights against the fp32 attention mask.
  - UNet only cast: UNet `time_proj/Mul` binds fp16 weights against
    the fp32 timestep input.

Switched to ORT's SD-aware path —
`onnxruntime.transformers.optimizer.optimize_model(model_type="unet")`
+ `convert_float_to_float16(keep_io_types=True)` — that produces a
working fp16 graph (UNet 3.3 GB → 1.7 GB, total 2.4 GB) but the
SD-specific fusions emit Microsoft contrib ops (NhwcConv, GroupNorm,
etc.) for which the desktop CPUExecutionProvider has no kernel:
    NOT_IMPLEMENTED: Failed to find kernel for com.microsoft.GroupNorm(1)
Disabling individual fusions via FusionOptions makes each subsequent
contrib op surface in turn. ORT's SD optimizer assumes a CUDA EP.

Decision: ship Step 2's fp32 ONNX as-is. fp16 cast deferred to
Step 5 (Kotlin/Android side) where we control ORT InferenceSession
directly — there we can pass fp16 IO tensors, use NNAPI / QNN
delegates that DO have GroupNorm kernels (Hexagon NPU), or rely on
session-init runtime optimization. This script + the verify script
stay so the next retry already has the failure surface mapped.

scheduler/, tokenizer/, model_index.json are not ONNX — straight copy.

The block of code below is the last attempt configuration (UNet
only via onnxconverter_common). Running it fills models/onnx_fp16/
with a directory that *cannot* be loaded by OnnxStableDiffusionPipeline
on CPU EP — same time_proj/Mul type-binding error described above.

keep_io_types=True so the I/O tensors stay fp32. Phone-side runtime
can feed fp32 tensors and ORT will do the cast at the boundary.
This makes the model a drop-in replacement for the fp32 version in
OnnxStableDiffusionPipeline; cheap I/O cast vs forking the pipeline.
"""
from __future__ import annotations

import sys
import shutil
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models" / "onnx"      / "sd15_drawing_nty_scale0.8"
DST_DIR = ROOT / "models" / "onnx_fp16" / "sd15_drawing_nty_scale0.8"

FP16_SUBDIRS = ["unet"]                                            # safe to cast
FP32_KEEP    = ["text_encoder", "vae_encoder", "vae_decoder"]      # crash on cast — copy as-is
COPY_SUBDIRS = ["scheduler", "tokenizer"]
COPY_FILES   = ["model_index.json"]


def cast_one(name: str, external_data: bool) -> None:
    src = SRC_DIR / name / "model.onnx"
    dst = DST_DIR / name / "model.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fp16] {name}: loading from {src}")
    t = time.time()
    model = onnx.load(str(src), load_external_data=True)
    print(f"[fp16]   loaded in {time.time() - t:.1f}s, casting to fp16 ...")
    t = time.time()
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,  # SD UNet has dynamic shapes that confuse infer
    )
    print(f"[fp16]   cast in {time.time() - t:.1f}s, saving to {dst}")
    t = time.time()
    if external_data:
        # UNet > 2 GB protobuf cap — split weights into a sidecar file.
        onnx.save(
            fp16_model,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            convert_attribute=False,
        )
    else:
        onnx.save(fp16_model, str(dst))
    print(f"[fp16]   saved in {time.time() - t:.1f}s")


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"missing source ONNX dir — run Step 2 first: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    cast_one("unet", external_data=True)

    # Copy fp32 sub-models as-is (VAE encoder/decoder).
    for sub in FP32_KEEP:
        src = SRC_DIR / sub
        dst = DST_DIR / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[fp16] kept {sub} at fp32 (copied)")

    for sub in COPY_SUBDIRS:
        src = SRC_DIR / sub
        dst = DST_DIR / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for fname in COPY_FILES:
        shutil.copy(SRC_DIR / fname, DST_DIR / fname)

    total = sum(f.stat().st_size for f in DST_DIR.rglob("*") if f.is_file())
    src_total = sum(f.stat().st_size for f in SRC_DIR.rglob("*") if f.is_file())
    ratio = total / src_total if src_total else 0
    print(f"\n[fp16] DONE")
    print(f"  fp32 src: {src_total / 1_000_000_000:.2f} GB")
    print(f"  fp16 dst: {total / 1_000_000_000:.2f} GB ({ratio*100:.0f}% of fp32)")
    for sub in sorted(DST_DIR.iterdir()):
        if sub.is_dir():
            sub_size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            print(f"  {sub.name:20s} {sub_size / 1_000_000:8.1f} MB")


if __name__ == "__main__":
    main()
