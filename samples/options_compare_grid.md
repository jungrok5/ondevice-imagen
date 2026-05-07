# Lightning + style-LoRA quality fixes — A/B/C/D shootout

Same Christmas event, same prompt, same seed (42), same style LoRA
(worstimever). Only the *speed-distillation method* differs per row.

Data phrases injected: `evening dusk, lamps lit, winter, bare branches, Christmas day, fairy lights, festive, clear sky, Korean setting, tall apartment buildings`

| option | result |
| --- | --- |
| B — Lightning 8-step + worst@0.9 | ![](opt_B_lightning_8step.png) |
| C — Lightning 4-step + worst@1.2 | ![](opt_C_worst_scale12.png) |
| D — Hyper-SD 4-step + worst@0.9 | ![](opt_D_hyper.png) |
| A — Lightning UNet variant + worst@0.9 (no LoRA conflict) | ![](opt_A_lightning_unet.png) |
