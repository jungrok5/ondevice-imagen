# Randomness check — same event × 4 calls

Same exact `EVENT` dict, same LoRA. Each call to
`build_event_prompt()` draws different phrases from the pools
in `src/event_prompt.py`. Result: 4 different images from
identical input.

Diffusion seed varies per cell: `SAME_DIFFUSION_SEED=False`

| # | data phrases (post-trigger, pre-style-tags) | result |
| --- | --- | --- |
| 0 | evening dusk, lamps lit, deep blue sky, wintertime, dry brown grass, gray cold sky, Christmas, snowflakes falling | ![](rand_0.png) |
| 1 | early night, glowing windows, street lamps on, wintertime, dry brown grass, gray cold sky, Christmas, red-and-gre | ![](rand_1.png) |
| 2 | early night, glowing windows, street lamps on, deep winter, leafless trees, icy ground, Christmas, red-and-green  | ![](rand_2.png) |
| 3 | early night, glowing windows, street lamps on, deep winter, leafless trees, icy ground, Christmas day, fairy ligh | ![](rand_3.png) |
