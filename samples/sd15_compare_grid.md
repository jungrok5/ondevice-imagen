# SD 1.5 LoRA compare — Drawing 110244 + Inked Portrait 947591

Same 5 events as `lora_catalog.md`. Base: `stable-diffusion-v1-5/stable-diffusion-v1-5`
(SD 1.5 vanilla, not Turbo — these LoRAs are SD 1.5 only).

Sampler: DPM++ 2M Karras, **12 steps**, CFG **7.0**, 
512x512, fixed seed **42**, `prompt_seed=7` so each LoRA sees the same data prompt.

Drawing 110244 is fused at scale 0.8 (mirrors the Civitai example
image the user picked); Inked Portrait 947591 at 0.9 (no author
guidance, matching the SDXL catalog's default).

| event | data |
| --- | --- |
| `e1_morning_home_livingroom` | 08:00 / 맑음 / 2026-04-12 / 거실 |
| `e2_evening_cafe_rain` | 20:00 / 비 / 2026-10-08 / 광화문 카페 |
| `e3_summer_river_night` | 22:00 / 맑음 / 2026-07-20 / 한강공원 |
| `e4_winter_market_noon` | 12:30 / 눈 / 2026-01-15 / 광장시장 |
| `e5_indoor_library_afternoon` | 15:00 / 흐림 / 2026-09-25 / 도서관 |

## sd15_drawing_nty  (trigger: `(style by NTY, drawing:1.2)`, scale 0.8)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](sd15_compare_sd15_drawing_nty__e1_morning_home_livingroom.png) | ![](sd15_compare_sd15_drawing_nty__e2_evening_cafe_rain.png) | ![](sd15_compare_sd15_drawing_nty__e3_summer_river_night.png) | ![](sd15_compare_sd15_drawing_nty__e4_winter_market_noon.png) | ![](sd15_compare_sd15_drawing_nty__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens):

- **`e1_morning_home_livingroom`** — (style by NTY, drawing:1.2), morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — (style by NTY, drawing:1.2), twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — (style by NTY, drawing:1.2), midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — (style by NTY, drawing:1.2), midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — (style by NTY, drawing:1.2), late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## sd15_inked_portrait  (trigger: `Illustration, portait style, black and white, line, ink, sketch`, scale 0.9)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](sd15_compare_sd15_inked_portrait__e1_morning_home_livingroom.png) | ![](sd15_compare_sd15_inked_portrait__e2_evening_cafe_rain.png) | ![](sd15_compare_sd15_inked_portrait__e3_summer_river_night.png) | ![](sd15_compare_sd15_inked_portrait__e4_winter_market_noon.png) | ![](sd15_compare_sd15_inked_portrait__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens):

- **`e1_morning_home_livingroom`** — Illustration, portait style, black and white, line, ink, sketch, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — Illustration, portait style, black and white, line, ink, sketch, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — Illustration, portait style, black and white, line, ink, sketch, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — Illustration, portait style, black and white, line, ink, sketch, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — Illustration, portait style, black and white, line, ink, sketch, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

