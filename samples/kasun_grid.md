# Kasun (한심함) filter ladder

Pure pixel-op post-process to push any image toward the 1-bit
MS Paint feel of the original 하찮은 프롬프트 GPT-4o output.

Levels:
- **light** : 64×64 grid, threshold 200 → cleaner doodle
- **medium**: 48×48 grid, threshold 180 → chunky bitmap
- **heavy** : 32×32 grid, threshold 180 → MS Paint
- **extreme**: 24×24 grid, threshold 180, +1 px row jitter (mouse tremor)

All outputs are 1024×1024 with nearest-neighbor upscale so the
blocks read clearly.

| source | light | medium | heavy | extreme |
| --- | --- | --- | --- | --- |
| ![](random_w11_doodle.png) | ![](kasun_light_random_w11_doodle.png) | ![](kasun_medium_random_w11_doodle.png) | ![](kasun_heavy_random_w11_doodle.png) | ![](kasun_extreme_random_w11_doodle.png) |
| ![](random_w11_txt2img.png) | ![](kasun_light_random_w11_txt2img.png) | ![](kasun_medium_random_w11_txt2img.png) | ![](kasun_heavy_random_w11_txt2img.png) | ![](kasun_extreme_random_w11_txt2img.png) |
| ![](cross_dreamlike_diffusion.png) | ![](kasun_light_cross_dreamlike_diffusion.png) | ![](kasun_medium_cross_dreamlike_diffusion.png) | ![](kasun_heavy_cross_dreamlike_diffusion.png) | ![](kasun_extreme_cross_dreamlike_diffusion.png) |
| ![](quality_realistic.png) | ![](kasun_light_quality_realistic.png) | ![](kasun_medium_quality_realistic.png) | ![](kasun_heavy_quality_realistic.png) | ![](kasun_extreme_quality_realistic.png) |
| ![](viral_friends_mom_portrait.png) | ![](kasun_light_viral_friends_mom_portrait.png) | ![](kasun_medium_viral_friends_mom_portrait.png) | ![](kasun_heavy_viral_friends_mom_portrait.png) | ![](kasun_extreme_viral_friends_mom_portrait.png) |
