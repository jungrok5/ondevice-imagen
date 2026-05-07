# v10 — SDXL-Turbo + worstimever LoRA, txt2img only

Product flow: diary text -> prompt_builder -> SDXL-Turbo + LoRA
-> result. No photo input. Seed is random — same data twice gives
different unexpected outputs.

| seed | week shape | prompt | output |
| --- | --- | --- | --- |
| w11_v1 | seed=11 weather | `ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scri...` | ![](v10_w11_v1.png) |
| w22_v1 | seed=22 place | `ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scri...` | ![](v10_w22_v1.png) |
| w33_v1 | seed=33 time | `ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scri...` | ![](v10_w33_v1.png) |
| w44_v1 | seed=44 cups | `ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scri...` | ![](v10_w44_v1.png) |
| w22_v2 | seed=22 place | `ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scri...` | ![](v10_w22_v2.png) |
