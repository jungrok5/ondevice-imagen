"""Phase 3 Step 3b — int8 quantization of the SDXL UNet.

Why int8 and not another fp16 attempt: the SDXL UNet is 2.6B params
(~10.3 GB fp32, ~5.1 GB fp16). Even a clean fp16 would not leave
headroom on 8 GB phones once TE1+TE2+VAE and the OS are resident —
Phase 2's lowmemorykiller fight (README §pitfall 3) taught us peak
RSS is the whole game. int8 puts the UNet at roughly 2.6 GB, and
Bonsai-style sub-2-bit work shows the quality headroom exists far
below int8, so int8 is the conservative point on that curve.

Why the UNet only: Phase 2 already established INT8 degrades the VAE
decoder badly (noise / banding) — VAE stays fp32/fp16. The text
encoders run once per image (~0.3 s) — quantizing them buys little
and risks prompt fidelity; skip. All the runtime lives in the UNet.

Two modes (MODE knob):

  "dynamic"  onnxruntime quantize_dynamic, MatMul weights only.
             No calibration needed. SDXL's params are transformer-
             heavy so MatMul covers most of the size. Activations
             quantized on the fly (CPU/XNNPACK path). The low-risk
             first shot — run this before bothering with QDQ.

  "qdq"      quantize_static in QDQ format with the calibration
             tensors captured by capture_unet_calib_phase3.py.
             QDQ is what the NNAPI / QNN EPs consume — this is the
             mode that unlocks Hexagon/NPU offload (README Step 6 /
             next-move C), where the projected UNet step drops from
             ~25 s (S25 CPU) toward the low single digits.

Quality gate, same discipline as every grid in this repo: regenerate
the 5 README events through the int8 UNet and eyeball against the
fp32 cells. worstimever's register (thick outlines, saturated fills)
is a sensitive canary — banding or washed fills = calibration or
per-channel settings need work, don't ship.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# === knobs ===
FUSED_NAME = "sdxl_turbo_worstimever_scale0.9"
MODE       = "dynamic"          # "dynamic" | "qdq"
PER_CHANNEL = True              # per-channel weights — matters at SDXL scale

ONNX_DIR  = ROOT / "models" / "onnx_xl" / FUSED_NAME
CALIB_DIR = ROOT / "models" / "calib_xl" / FUSED_NAME
DST_DIR   = ROOT / "models" / f"onnx_xl_int8_{MODE}" / FUSED_NAME

SRC_UNET = ONNX_DIR / "unet" / "model.onnx"
DST_UNET = DST_DIR / "unet" / "model.onnx"


def quantize_dynamic_mode() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("[int8] dynamic mode — MatMul weights to int8, no calibration")
    t = time.time()
    quantize_dynamic(
        model_input=str(SRC_UNET),
        model_output=str(DST_UNET),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=PER_CHANNEL,
        use_external_data_format=True,   # UNet > 2 GB protobuf cap
    )
    print(f"[int8]   quantized in {time.time() - t:.0f}s")


class NpzCalibrationReader:
    """Feeds capture_unet_calib_phase3.py's npz dumps to the calibrator.

    npz keys were written to match the optimum SDXL UNet input names
    1:1, so this is a straight replay. Extra graph inputs the export
    may add (e.g. timestep_cond) would surface here as a KeyError —
    that's a signal to re-run the capture script, not to fake data.
    """

    def __init__(self, calib_dir: Path):
        self.files = sorted(calib_dir.glob("sample_*.npz"))
        if not self.files:
            raise SystemExit(
                f"no calibration samples in {calib_dir} — run "
                "scripts/capture_unet_calib_phase3.py first"
            )
        self._it = iter(self.files)

    def get_next(self):
        import numpy as np
        path = next(self._it, None)
        if path is None:
            return None
        data = np.load(path)
        return {k: data[k] for k in data.files}


def quantize_qdq_mode() -> None:
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )

    reader = NpzCalibrationReader(CALIB_DIR)
    print(f"[int8] qdq mode — {len(reader.files)} calibration samples")
    t = time.time()
    quantize_static(
        model_input=str(SRC_UNET),
        model_output=str(DST_UNET),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,            # what NNAPI/QNN EPs read
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=PER_CHANNEL,
        op_types_to_quantize=["MatMul", "Conv"],
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
    )
    print(f"[int8]   quantized in {time.time() - t:.0f}s")


def main() -> None:
    if not SRC_UNET.exists():
        raise SystemExit(
            f"missing ONNX UNet — run Step 2 first "
            f"(scripts/export_sdxl_onnx_phase3.py): {SRC_UNET}"
        )
    DST_UNET.parent.mkdir(parents=True, exist_ok=True)

    if MODE == "dynamic":
        quantize_dynamic_mode()
    elif MODE == "qdq":
        quantize_qdq_mode()
    else:
        raise SystemExit(f"unknown MODE: {MODE}")

    src_sz = sum(
        f.stat().st_size for f in (ONNX_DIR / "unet").rglob("*") if f.is_file()
    )
    dst_sz = sum(
        f.stat().st_size for f in DST_UNET.parent.rglob("*") if f.is_file()
    )
    print(f"\n[int8] DONE ({MODE})")
    print(f"  fp32 unet: {src_sz / 1_000_000_000:.2f} GB")
    print(f"  int8 unet: {dst_sz / 1_000_000_000:.2f} GB "
          f"({dst_sz / src_sz * 100:.0f}% of fp32)")
    print(
        "\n[int8] other sub-models are NOT copied here — bundle "
        "text encoders / VAE from the Step 2 dir (VAE must stay "
        "fp32/fp16 per the Phase 2 finding)."
    )
    print(
        "[int8] quality gate: regenerate the 5 README events through "
        "this UNet and diff against the fp32 grid before pushing to "
        "the phone."
    )


if __name__ == "__main__":
    main()
