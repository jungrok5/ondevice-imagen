"""Phase 2 Step 5b-4a — single UNet forward pass reference.

Feeds a deterministic input (sample = all 0.5, timestep = 999, hidden
states from "a cat" prompt) through the same fp32 ONNX UNet the
phone uses, and dumps shape + summary stats + first 8 output values.
Phone-side InferenceForegroundService.probeUNet must match within
ORT's ~1e-4 numerical drift on identical inputs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models" / "onnx" / "sd15_drawing_nty_scale0.8"

PROMPT = "a cat"


def main() -> None:
    print(f"[ref] tokenizing '{PROMPT}'")
    tok = CLIPTokenizer.from_pretrained(str(ONNX_DIR / "tokenizer"))
    ids = tok(PROMPT, padding="max_length", max_length=77,
              truncation=True, return_tensors="np").input_ids[0].astype(np.int32)
    print(f"[ref] ids[0..7]={ids[:8].tolist()}")

    print(f"[ref] running text_encoder...")
    te = ort.InferenceSession(
        str(ONNX_DIR / "text_encoder" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    hidden = te.run(None, {"input_ids": ids[None, :]})[0]   # [1, 77, 768]
    print(f"[ref] hidden shape={hidden.shape} mean={hidden.mean():.5f}")

    print(f"[ref] loading UNet...")
    unet = ort.InferenceSession(
        str(ONNX_DIR / "unet" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    sample = np.full((1, 4, 64, 64), 0.5, dtype=np.float32)
    timestep = np.array([999.0], dtype=np.float32)

    print(f"[ref] running UNet (sample=all 0.5, timestep=999)...")
    t0 = time.time()
    out = unet.run(None, {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": hidden,
    })
    elapsed_ms = int((time.time() - t0) * 1000)
    out_sample = out[0]
    print(f"[ref] elapsed={elapsed_ms} ms shape={'x'.join(map(str, out_sample.shape))}")
    print(f"[ref] stats mean={out_sample.mean():.5f} std={out_sample.std():.5f} "
          f"min={out_sample.min():.5f} max={out_sample.max():.5f}")
    print(f"[ref] out[0,0,0,0..7]=" +
          ",".join(f"{v:.4f}" for v in out_sample[0, 0, 0, :8]))


if __name__ == "__main__":
    main()
