"""Phase 2 Step 5b-2 — produce reference text_encoder outputs.

Loads the same fp32 ONNX text_encoder via onnxruntime that the phone
uses, runs it on a known prompt, prints the same shape + summary
stats + fingerprint values that InferenceForegroundService.probeTextEncoder
logs from the phone. Numerical drift between PC ORT and phone ORT
should be tiny (~1e-5) — same model file, same opset, same EP.
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

SAMPLE_PROMPT = (
    "(style by NTY, drawing:1.2), a scene at 명동, evening, "
    "overcast weather, during spring, (2026-05-09 18:54)"
)


def main() -> None:
    print(f"[ref] tokenizing: '{SAMPLE_PROMPT[:60]}...'")
    tok = CLIPTokenizer.from_pretrained(str(ONNX_DIR / "tokenizer"))
    out = tok(
        SAMPLE_PROMPT, padding="max_length", max_length=77,
        truncation=True, return_tensors="np",
    )
    ids = out.input_ids[0].astype(np.int32)
    print(f"[ref] ids[0..7]={ids[:8].tolist()} ids[-3..]={ids[-3:].tolist()}")

    print(f"[ref] loading text_encoder ONNX...")
    sess = ort.InferenceSession(
        str(ONNX_DIR / "text_encoder" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    t0 = time.time()
    res = sess.run(None, {"input_ids": ids[None, :]})  # batch dim
    elapsed_ms = int((time.time() - t0) * 1000)
    hidden = res[0]                                      # last_hidden_state
    print(f"[ref] shape={'x'.join(map(str, hidden.shape))} elapsed={elapsed_ms} ms")
    print(f"[ref] stats mean={hidden.mean():.5f} std={hidden.std():.5f} "
          f"min={hidden.min():.5f} max={hidden.max():.5f}")
    print(f"[ref] hid[0,0,0..7]=" +
          ",".join(f"{v:.4f}" for v in hidden[0, 0, :8]))
    print(f"[ref] hid[0,1,0..7]=" +
          ",".join(f"{v:.4f}" for v in hidden[0, 1, :8]))


if __name__ == "__main__":
    main()
