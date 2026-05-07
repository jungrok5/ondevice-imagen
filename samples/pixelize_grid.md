# Pixelize gallery

Existing colorful samples remapped through `src/pixelize.py`. No SD
inference — pure PIL post-process.

- `pixel16_*.png`: 64×64 grid, 16-color median-cut palette picked per
  image. Preserves source colors with a retro paint-program feel.
- `pixelgb_*.png`: 64×64 grid, fixed 4-color Game Boy DMG-01 palette.
  Forces a 1989 game-screen aesthetic regardless of source colors.

| source | 16-color | Game Boy |
| --- | --- | --- |
| ![](drill_mom_korean_living_room.png) | ![](pixel16_drill_mom_korean_living_room.png) | ![](pixelgb_drill_mom_korean_living_room.png) |
| ![](quality_realistic.png) | ![](pixel16_quality_realistic.png) | ![](pixelgb_quality_realistic.png) |
| ![](quality_anime.png) | ![](pixel16_quality_anime.png) | ![](pixelgb_quality_anime.png) |
| ![](style_watercolor_diary.png) | ![](pixel16_style_watercolor_diary.png) | ![](pixelgb_style_watercolor_diary.png) |
| ![](cross_dreamlike_diffusion.png) | ![](pixel16_cross_dreamlike_diffusion.png) | ![](pixelgb_cross_dreamlike_diffusion.png) |
| ![](viral_hotel_art_70s.png) | ![](pixel16_viral_hotel_art_70s.png) | ![](pixelgb_viral_hotel_art_70s.png) |
| ![](viral_friends_mom_portrait.png) | ![](pixel16_viral_friends_mom_portrait.png) | ![](pixelgb_viral_friends_mom_portrait.png) |
