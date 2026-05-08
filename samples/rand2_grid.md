# Two-LoRA randomness check — 4 calls each

Same `EVENT` dict, both style LoRAs (`worstimever`,
`mspaint_portraits`), 4 calls per LoRA. Each call draws
fresh phrases from the pools in `src/event_prompt.py`. The
diffusion seed also varies per cell so prompt + denoise
randomness combine.

| call # | worstimever | mspaint_portraits |
| --- | --- | --- |
| 0 | ![](rand2_worstimever_0.png) | ![](rand2_mspaint_portraits_0.png) |
| 1 | ![](rand2_worstimever_1.png) | ![](rand2_mspaint_portraits_1.png) |
| 2 | ![](rand2_worstimever_2.png) | ![](rand2_mspaint_portraits_2.png) |
| 3 | ![](rand2_worstimever_3.png) | ![](rand2_mspaint_portraits_3.png) |
