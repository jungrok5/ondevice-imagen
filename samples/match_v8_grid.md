# v8 — prompt variation sweep on the v7 pipeline

Same pipeline as v7 (img2img s=0.75 -> cv2.stylization -> row-jitter ±3 px). Only the prompt wording differs.

| variant | prompt | image |
| --- | --- | --- |
| input | — | ![](reference/user_test/line/set1_input.png) |
| p1_child_mouse | `drawn by a five-year-old child with a computer mouse, awkwar...` | ![](v8_p1_child_mouse_final.png) |
| p2_wrong_hand | `sketched in five seconds with the wrong hand, terrible mouse...` | ![](v8_p2_wrong_hand_final.png) |
| p3_ms_paint_1995 | `Microsoft Paint amateur drawing from 1995, crude pixel-mouse...` | ![](v8_p3_ms_paint_1995_final.png) |
| p4_kindergarten_crayon | `kindergarten crayon drawing on white paper, naive figure, si...` | ![](v8_p4_kindergarten_crayon_final.png) |
| p5_intentionally_bad | `intentionally awful sketch, deliberately badly drawn, ugly p...` | ![](v8_p5_intentionally_bad_final.png) |
| **target — ChatGPT 하찮은** | — | ![](reference/user_test/line/set1_output.png) |
| user-favourite — v5 s=0.85 | — | ![](v5_set1_sd_s85.png) |
