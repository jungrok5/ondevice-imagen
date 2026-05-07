# v9 — SD-Turbo + COOLKIDS LoRA + cv2 stylize + jitter

Adds the COOLKIDS V2 LoRA (Clumsy_Trainer, SD 1.5 compatible,
144 MB) so SD-Turbo natively produces children-book illustration
without needing as heavy post-processing.

| variant | image |
| --- | --- |
| input | ![](reference/user_test/line/set1_input.png) |
| LoRA scale=0.6 (raw SD) | ![](v9_s06_sd.png) |
| LoRA scale=0.6 + cv2 (no jitter) | ![](v9_s06_final.png) |
| LoRA scale=0.9 (raw SD) | ![](v9_s09_sd.png) |
| LoRA scale=0.9 + cv2 (no jitter) | ![](v9_s09_final.png) |
| target — ChatGPT 하찮은 | ![](reference/user_test/line/set1_output.png) |
| user-favourite — v5 s=0.85 | ![](v5_set1_sd_s85.png) |
