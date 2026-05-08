# LoRA catalog — diverse events × multiple style LoRAs

Same SDXL-Turbo, same diffusion seed (42), same `prompt_seed=7`
(so all LoRAs see the same data prompt for a given event). Each
LoRA fused at scale 0.9 with its Civitai-official trigger
auto-prepended via `LORA_TRIGGERS`.

Five events span indoor and outdoor, four seasons, four weather
types, morning through night.

| event | data |
| --- | --- |
| `e1_morning_home_livingroom` | 08:00 / 맑음 / 2026-04-12 / 거실 |
| `e2_evening_cafe_rain` | 20:00 / 비 / 2026-10-08 / 광화문 카페 |
| `e3_summer_river_night` | 22:00 / 맑음 / 2026-07-20 / 한강공원 |
| `e4_winter_market_noon` | 12:30 / 눈 / 2026-01-15 / 광장시장 |
| `e5_indoor_library_afternoon` | 15:00 / 흐림 / 2026-09-25 / 도서관 |

## worstimever  (trigger: `WTE artstyle`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_worstimever__e1_morning_home_livingroom.png) | ![](cat_worstimever__e2_evening_cafe_rain.png) | ![](cat_worstimever__e3_summer_river_night.png) | ![](cat_worstimever__e4_winter_market_noon.png) | ![](cat_worstimever__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — WTE artstyle, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — WTE artstyle, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — WTE artstyle, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — WTE artstyle, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — WTE artstyle, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## mspaint_portraits  (trigger: `MSPaint drawing`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_mspaint_portraits__e1_morning_home_livingroom.png) | ![](cat_mspaint_portraits__e2_evening_cafe_rain.png) | ![](cat_mspaint_portraits__e3_summer_river_night.png) | ![](cat_mspaint_portraits__e4_winter_market_noon.png) | ![](cat_mspaint_portraits__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — MSPaint drawing, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — MSPaint drawing, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — MSPaint drawing, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — MSPaint drawing, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — MSPaint drawing, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## doodle_style  (trigger: `SDXL_BTT_Doodle_v01`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_doodle_style__e1_morning_home_livingroom.png) | ![](cat_doodle_style__e2_evening_cafe_rain.png) | ![](cat_doodle_style__e3_summer_river_night.png) | ![](cat_doodle_style__e4_winter_market_noon.png) | ![](cat_doodle_style__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — SDXL_BTT_Doodle_v01, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — SDXL_BTT_Doodle_v01, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — SDXL_BTT_Doodle_v01, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — SDXL_BTT_Doodle_v01, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — SDXL_BTT_Doodle_v01, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## doodles_in_real_life  (trigger: `photo doodle`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_doodles_in_real_life__e1_morning_home_livingroom.png) | ![](cat_doodles_in_real_life__e2_evening_cafe_rain.png) | ![](cat_doodles_in_real_life__e3_summer_river_night.png) | ![](cat_doodles_in_real_life__e4_winter_market_noon.png) | ![](cat_doodles_in_real_life__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — photo doodle, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — photo doodle, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — photo doodle, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — photo doodle, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — photo doodle, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## cyberpunk_lines  (trigger: `lineAnime`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_cyberpunk_lines__e1_morning_home_livingroom.png) | ![](cat_cyberpunk_lines__e2_evening_cafe_rain.png) | ![](cat_cyberpunk_lines__e3_summer_river_night.png) | ![](cat_cyberpunk_lines__e4_winter_market_noon.png) | ![](cat_cyberpunk_lines__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — lineAnime, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — lineAnime, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — lineAnime, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — lineAnime, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — lineAnime, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## linedrawing  (trigger: `LineDrawing(style)`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_linedrawing__e1_morning_home_livingroom.png) | ![](cat_linedrawing__e2_evening_cafe_rain.png) | ![](cat_linedrawing__e3_summer_river_night.png) | ![](cat_linedrawing__e4_winter_market_noon.png) | ![](cat_linedrawing__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — LineDrawing(style), morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — LineDrawing(style), twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — LineDrawing(style), midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — LineDrawing(style), midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — LineDrawing(style), late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## lineart_zoolin  (trigger: `lineart style`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| (failed) | (failed) | (failed) | (failed) | (failed) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — lineart style, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — lineart style, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — lineart style, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — lineart style, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — lineart style, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## asian_line_storyboard  (trigger: `asian, storyboard, sketch`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_asian_line_storyboard__e1_morning_home_livingroom.png) | ![](cat_asian_line_storyboard__e2_evening_cafe_rain.png) | ![](cat_asian_line_storyboard__e3_summer_river_night.png) | ![](cat_asian_line_storyboard__e4_winter_market_noon.png) | ![](cat_asian_line_storyboard__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — asian, storyboard, sketch, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — asian, storyboard, sketch, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — asian, storyboard, sketch, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — asian, storyboard, sketch, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — asian, storyboard, sketch, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## soft_squishy_linework  (trigger: `<none>`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_soft_squishy_linework__e1_morning_home_livingroom.png) | ![](cat_soft_squishy_linework__e2_evening_cafe_rain.png) | ![](cat_soft_squishy_linework__e3_summer_river_night.png) | ![](cat_soft_squishy_linework__e4_winter_market_noon.png) | ![](cat_soft_squishy_linework__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## colored_line  (trigger: `SCTX`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| (failed) | (failed) | (failed) | (failed) | (failed) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — SCTX, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — SCTX, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — SCTX, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — SCTX, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — SCTX, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## clean_bw_line_art  (trigger: `K3NJIKUN ARTSTYLE`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_clean_bw_line_art__e1_morning_home_livingroom.png) | ![](cat_clean_bw_line_art__e2_evening_cafe_rain.png) | ![](cat_clean_bw_line_art__e3_summer_river_night.png) | ![](cat_clean_bw_line_art__e4_winter_market_noon.png) | ![](cat_clean_bw_line_art__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — K3NJIKUN ARTSTYLE, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — K3NJIKUN ARTSTYLE, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — K3NJIKUN ARTSTYLE, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — K3NJIKUN ARTSTYLE, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — K3NJIKUN ARTSTYLE, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## tangbohu_landscape  (trigger: `<none>`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_tangbohu_landscape__e1_morning_home_livingroom.png) | ![](cat_tangbohu_landscape__e2_evening_cafe_rain.png) | ![](cat_tangbohu_landscape__e3_summer_river_night.png) | ![](cat_tangbohu_landscape__e4_winter_market_noon.png) | ![](cat_tangbohu_landscape__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## digital_art_illustrations  (trigger: `J_GUOCHAO`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_digital_art_illustrations__e1_morning_home_livingroom.png) | ![](cat_digital_art_illustrations__e2_evening_cafe_rain.png) | ![](cat_digital_art_illustrations__e3_summer_river_night.png) | ![](cat_digital_art_illustrations__e4_winter_market_noon.png) | ![](cat_digital_art_illustrations__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — J_GUOCHAO, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — J_GUOCHAO, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — J_GUOCHAO, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — J_GUOCHAO, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — J_GUOCHAO, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## simplex  (trigger: `SIMPLEX`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_simplex__e1_morning_home_livingroom.png) | ![](cat_simplex__e2_evening_cafe_rain.png) | ![](cat_simplex__e3_summer_river_night.png) | ![](cat_simplex__e4_winter_market_noon.png) | ![](cat_simplex__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — SIMPLEX, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — SIMPLEX, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — SIMPLEX, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — SIMPLEX, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — SIMPLEX, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## simple_toons  (trigger: `a simple cartoon illustration, clean lineart`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_simple_toons__e1_morning_home_livingroom.png) | ![](cat_simple_toons__e2_evening_cafe_rain.png) | ![](cat_simple_toons__e3_summer_river_night.png) | ![](cat_simple_toons__e4_winter_market_noon.png) | ![](cat_simple_toons__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — a simple cartoon illustration, clean lineart, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — a simple cartoon illustration, clean lineart, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — a simple cartoon illustration, clean lineart, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — a simple cartoon illustration, clean lineart, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — a simple cartoon illustration, clean lineart, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## jackledead_artstyle  (trigger: `POP ART, DRAWING`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_jackledead_artstyle__e1_morning_home_livingroom.png) | ![](cat_jackledead_artstyle__e2_evening_cafe_rain.png) | ![](cat_jackledead_artstyle__e3_summer_river_night.png) | ![](cat_jackledead_artstyle__e4_winter_market_noon.png) | ![](cat_jackledead_artstyle__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — POP ART, DRAWING, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — POP ART, DRAWING, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — POP ART, DRAWING, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — POP ART, DRAWING, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — POP ART, DRAWING, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

## japanese_illustration  (trigger: `AKINO`)

| e1_morning_home_livingroom | e2_evening_cafe_rain | e3_summer_river_night | e4_winter_market_noon | e5_indoor_library_afternoon |
| --- | --- | --- | --- | --- |
| ![](cat_japanese_illustration__e1_morning_home_livingroom.png) | ![](cat_japanese_illustration__e2_evening_cafe_rain.png) | ![](cat_japanese_illustration__e3_summer_river_night.png) | ![](cat_japanese_illustration__e4_winter_market_noon.png) | ![](cat_japanese_illustration__e5_indoor_library_afternoon.png) |

**Prompts** (CLIP truncates at 77 tokens; tail beyond is dropped):

- **`e1_morning_home_livingroom`** — AKINO, morning light, crisp clean atmosphere, fresh start of the day, early spring, pink and white blossoms on trees, light layered clothing, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, indoor home living room, sofa, low coffee table, warm floor lamp, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e2_evening_cafe_rain`** — AKINO, twilight, first stars appearing, warm lamp glow spilling onto pavement, late autumn, bare-ish branches, piles of orange leaves, drizzle, fine rain in the air, soaked sidewalks, Korean setting, indoor cozy Korean cafe interior, wooden tables, warm hanging lights, espresso machine on counter, Gwanghwamun, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e3_summer_river_night`** — AKINO, midnight, crescent moon, stars scattered, dark blue-black sky, only lamp glow, high summer, dense tree canopy, cicadas implied, t-shirts and shorts, sunny day, big white puffy clouds drifting, vivid contrast, Korean setting, Hangang Park by the river, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e4_winter_market_noon`** — AKINO, midday hour, bleaching overhead light, people on a quick lunch break, deep winter, leafless trees, icy ground, fluffy snow drifting down, snow hats on every surface, kids making snowman, Korean setting, Gwangjang Market, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing
- **`e5_indoor_library_afternoon`** — AKINO, late afternoon, long shadows, golden side light, late autumn, bare-ish branches, piles of orange leaves, heavy gray cloud cover, muted dull palette, soft uniform light, Korean setting, indoor quiet library interior, tall bookshelves, reading desks, soft warm lamp light, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing

