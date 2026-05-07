# v7 — cv2 stylization + row-jitter wobble

User-preferred direction (per v5_s85 feedback): colourful cartoon
with thick black outlines, full scene. v7 takes v6 stylize_60_045
and adds mouse-tremor wobble via per-row horizontal jitter.

| stage | image |
| --- | --- |
| input | ![](reference/user_test/line/set1_input.png) |
| SD-Turbo img2img (s=0.75) | ![](v7_set1_sd.png) |
| + cv2.stylization | ![](v7_set1_stylized.png) |
| + jitter ±1 px | ![](v7_set1_final_j1.png) |
| + jitter ±2 px | ![](v7_set1_final_j2.png) |
| + jitter ±3 px | ![](v7_set1_final_j3.png) |
| **target — ChatGPT 하찮은** | ![](reference/user_test/line/set1_output.png) |
| user-favourite — v5 s=0.85 | ![](v5_set1_sd_s85.png) |
