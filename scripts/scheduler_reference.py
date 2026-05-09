"""Phase 2 Step 5b-3 — emit reference values from
DPMSolverMultistepScheduler with the same config the phone-side
Kotlin port will use. Two artifacts:

  1. The full Karras sigma schedule (13 numbers for num_inference_steps=12
     — diffusers prepends 0 at the end).
  2. A single canned scheduler.step() — given a fixed dummy
     `model_output` and `sample`, what does the scheduler return?
     The phone Kotlin step() must match within 1e-5.

This pins the Kotlin port: any drift between the phone log and
this output flags a math bug before we wire it to UNet.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent
SCHED_DIR = ROOT / "models" / "onnx" / "sd15_drawing_nty_scale0.8" / "scheduler"

NUM_INFERENCE_STEPS = 12


def main() -> None:
    sched = DPMSolverMultistepScheduler.from_pretrained(str(SCHED_DIR))
    # sd15_lora_compare.py + Step 1 verify both used use_karras_sigmas=True
    sched.config.use_karras_sigmas = True
    sched.set_timesteps(NUM_INFERENCE_STEPS)

    print(f"[ref] config: {dict(sched.config)}")
    print()

    sigmas = sched.sigmas.cpu().numpy()
    timesteps = sched.timesteps.cpu().numpy()
    print(f"[ref] sigmas (n+1={len(sigmas)}):")
    for i, s in enumerate(sigmas):
        print(f"  sigma[{i:2d}] = {float(s):.6f}")
    print(f"\n[ref] timesteps (n={len(timesteps)}):")
    print(f"  {timesteps.tolist()}")

    # Canned step — feed deterministic inputs so the phone can replay.
    # Latent shape for SD 1.5 at 512x512 is [batch=1, ch=4, h=64, w=64].
    # We use a tiny [1,4,2,2] for log compactness while keeping math
    # the same. Step 5b-4's UNet loop uses the real shape.
    rng = np.random.RandomState(42)
    dummy_sample = rng.randn(1, 4, 2, 2).astype(np.float32)
    dummy_model_out = rng.randn(1, 4, 2, 2).astype(np.float32) * 0.5

    print(f"\n[ref] dummy_sample[0,0]:")
    print(f"  {dummy_sample[0, 0].flatten().tolist()}")
    print(f"[ref] dummy_model_out[0,0]:")
    print(f"  {dummy_model_out[0, 0].flatten().tolist()}")

    # Run a single step at index 0 (sigma=14.6146, the highest)
    # Reset solver state because diffusers caches between steps.
    sched.set_timesteps(NUM_INFERENCE_STEPS)
    sample_t = torch.tensor(dummy_sample)
    model_out_t = torch.tensor(dummy_model_out)
    timestep = sched.timesteps[0]
    out = sched.step(model_out_t, timestep, sample_t).prev_sample.numpy()

    print(f"\n[ref] step(timestep={int(timestep)}) -> prev_sample[0,0]:")
    print(f"  {out[0, 0].flatten().tolist()}")
    print(f"[ref] prev_sample stats: "
          f"mean={out.mean():.5f} std={out.std():.5f} "
          f"min={out.min():.5f} max={out.max():.5f}")


if __name__ == "__main__":
    main()
