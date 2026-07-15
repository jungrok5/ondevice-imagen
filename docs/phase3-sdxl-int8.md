# Phase 3 — SDXL-Turbo + int8 on the phone

Phase 2 proved the native pipeline: SD 1.5 fp32 renders a real PNG on
Android through our own Kotlin ORT stack. But the styles the README
grids actually selected — `worstimever`, `mspaint_portraits` — are
**SDXL LoRAs on SDXL-Turbo**, a model generation above what the phone
runs. Phase 3 carries the PC-verified combo to the device.

## Why this is the right next move (and not 1-bit / FLUX)

- **LoRAs and checkpoints are architecture-bound.** SD1.5 LoRAs fit
  SD1.5, SDXL LoRAs fit SDXL. Quantization doesn't "unlock better
  LoRAs" — it lets a *bigger base* fit, and the base brings its LoRA
  ecosystem with it. Our chosen LoRAs are already SDXL: the upgrade
  path is paved.
- **int8 is hardware-native on our targets.** NNAPI / QNN (Hexagon)
  consume QDQ int8 directly. Sub-2-bit (Bonsai/BitNet-style) needs
  custom low-bit GEMM kernels that don't exist off-the-shelf for
  Android — that's a research chapter, not this one.
- **Turbo's schedule is a mobile gift.** guidance 0.0 → no CFG →
  UNet batch=1 (Phase 2 ran batch=2), and 4 steps vs 12. That's 6x
  fewer UNet forwards than Phase 2 *before* quantization or NPU work.

## Memory math (to be measured, not trusted)

| sub-model | params | fp32 | fp16 | int8 (UNet only) |
|---|---|---|---|---|
| UNet | ~2.6B | ~10.3 GB | ~5.1 GB | **~2.6 GB** |
| TE1 (CLIP-L) | 123M | 0.5 GB | 0.25 GB | keep fp16/fp32 |
| TE2 (OpenCLIP bigG) | 695M | 2.8 GB | 1.4 GB | keep fp16/fp32 |
| VAE | 84M | 0.3 GB | 0.17 GB | **never int8** (Phase 2 finding: banding) |

Peak-RSS strategy on 8 GB devices: **stage the sub-models** — run
TE1+TE2 once, free them, keep only the int8 UNet resident for the
4-step loop, then load the VAE decoder. We own the pipeline
(SdInferencePipeline.kt), so sequential load/unload is a session
lifecycle change, not a framework fight. This is the capability that
canned runtimes don't give and Phase 2's LMK battle (README pitfall
3) says we need.

## Steps

| # | step | script | status |
|---|---|---|---|
| 1 | fuse `worstimever_xl` @0.9 into `sdxl-turbo`, fp16 save + fp32 verify cell | `scripts/merge_sdxl_lora_phase3.py` | ready to run (PC) |
| 2 | ONNX export via optimum (`stable-diffusion-xl` task) + optional ORT verify cell | `scripts/export_sdxl_onnx_phase3.py` | ready to run (PC, needs `pip install "optimum[exporters]"`) |
| 3a | capture 20 real UNet input tuples (5 events × 4 steps) for calibration | `scripts/capture_unet_calib_phase3.py` | ready to run (PC) |
| 3b | UNet int8 — `dynamic` (MatMul-only, no calib) first, then `qdq` (static, NNAPI food) | `scripts/quantize_sdxl_unet_int8_phase3.py` | ready to run (PC) |
| 4 | quality gate: 5 README events through int8 UNet vs fp32 grid, eyeball + commit cells | (reuse compare harness) | — |
| 5 | Kotlin pipeline: SDXL support (below) | `SdInferencePipeline.kt` etc. | design below |
| 6 | NNAPI / QNN EP with the QDQ UNet — the actual speed unlock | — | after 5 |

## Step 5 — Kotlin-side changes (what SDXL adds over SD 1.5)

1. **Second text encoder + tokenizer.** SDXL runs CLIP ViT-L (768)
   *and* OpenCLIP bigG (1280); the UNet takes their concatenated
   hidden states (77×2048) plus bigG's pooled output. `ClipTokenizer.kt`
   is byte-level BPE and reusable, but tokenizer_2 has a different
   vocab/pad configuration — verify token ids byte-identical against
   `transformers` for both tokenizers, same discipline as Phase 2 5b-1.
2. **Micro-conditioning inputs.** UNet gains `text_embeds` (1×1280)
   and `time_ids` (1×6 = original_size + crop + target_size). For our
   fixed 512×512 flow, time_ids is a constant `[512,512,0,0,512,512]`
   — compute once, feed every step.
3. **Scheduler port.** sdxl-turbo's default is EulerAncestral, which
   draws fresh noise *per step* (stochastic) — unlike the Euler-Karras
   in `DpmScheduler.kt`. Port `EulerAncestralDiscreteScheduler` with a
   seeded RNG and verify step outputs against a PC reference canned
   run, same method as 5b-3.
4. **No CFG.** guidance 0.0 removes the uncond pass — UNet batch=1.
   Delete the CFG mixing, halve the per-step cost for free.
5. **Sub-model staging.** Load TE1→run→close, TE2→run→close, UNet
   resident for 4 steps, close, VAE decode. Keeps peak RSS ≈ int8
   UNet + activations.
6. **Latent scaling constant.** SDXL VAE `scaling_factor` is 0.13025
   (SD1.5: 0.18215) — a silent-wrong-image constant if missed.

## Risks / open questions

- **Dynamic-quant CPU perf on ARM**: dynamic MatMul int8 helps memory
  for sure; CPU speed gain depends on ORT Mobile's XNNPACK int8 paths
  on our devices — measure, don't assume. The speed story is really
  Step 6 (QDQ + NNAPI/QNN).
- **QDQ op coverage on NNAPI**: GroupNorm/SiLU segments may fall back
  to CPU between NPU partitions; partition count is the metric to log
  (ORT session profiling), same instinct as Phase 2's schema probes.
- **Turbo + int8 interaction**: distilled models can be less tolerant
  of weight noise than base models. If `dynamic` already shows fill
  banding at the quality gate, jump to `qdq` per-channel before
  concluding int8 is off the table.
- **Download budget**: int8 UNet ~2.6 GB + TEs + VAE ≈ ~4.5 GB bundle
  — bigger than Phase 2's 4.27 GB was for SD1.5 fp32. fp16-casting
  the TEs (the Phase 2 Step 3 blocker was per-sub-model; TEs may cast
  clean via ORT transformers optimizer with `model_type="clip"`)
  would pull the bundle toward ~3.5 GB.
