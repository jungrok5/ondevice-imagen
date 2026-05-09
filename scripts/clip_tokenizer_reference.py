"""Phase 2 Step 5b-1 — produce reference CLIP tokenizer IDs.

Run this and compare against the Note 10+ logcat output from
InferenceForegroundService.probeTokenizer(). They must match
exactly: id-for-id, including the trailing EOS/PAD pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from transformers import CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "models" / "onnx" / "sd15_drawing_nty_scale0.8" / "tokenizer"

SAMPLES = [
    "a photo of a cat",
    "a cat",
    "(style by NTY, drawing:1.2)",
    # An example "live" prompt the phone might build:
    "(style by NTY, drawing:1.2), a scene at 명동, evening, "
    "overcast weather, during spring, (2026-05-09 18:54)",
]


def main() -> None:
    tok = CLIPTokenizer.from_pretrained(str(TOK_DIR))
    for s in SAMPLES:
        out = tok(
            s, padding="max_length", max_length=77,
            truncation=True, return_tensors="np",
        )
        ids = out.input_ids[0].tolist()
        # Trim trailing PAD (== EOS == 49407) for readability,
        # keep one EOS for context.
        last_non_pad = next(
            (i for i in range(len(ids) - 1, -1, -1) if ids[i] != 49407), 0
        )
        show = ids[:min(last_non_pad + 2, 24)]
        print(f"ref: '{s[:60]}' -> {','.join(map(str, show))} "
              f"({last_non_pad + 1} non-pad)")


if __name__ == "__main__":
    main()
