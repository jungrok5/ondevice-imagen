# Random weekly doodles

Four randomized weekly diaries fed through the same pipeline:

1. `make_random_week(seed)` synthesises a week of water/weather/places.
2. `prompt_builder.build()` turns it into an SD prompt + a deterministic
   pictographic collage (the img2img seed).
3. `generator.generate_img2img()` runs SD-Turbo at 4 steps, strength 0.99.
4. `pixelize()` enforces a true pixel-art look (64×64 grid, 16-color
   palette, nearest-neighbor upscale).

| seed | summary | subjects in prompt | collage | SD-Turbo doodle | pixel-perfect |
| --- | --- | --- | --- | --- | --- |
| 11 | Week 2026-04-27 ~ 2026-05-03 \| Total water: 5800 ml across 24 sips \| Avg/day: 828.6 ml \| Dominant weather: windy (avg 13.4 C) \| Place categories: {'cafe': 1, 'bookstore': 1, 'work': 1} \| Top places: Cafe, Bookstore, Workplace | 8 mismatched water cups arranged in a row; squiggly wind lines, leaves flying sideways; wobbly coffee cup with steam squiggles; stack of crooked books; ugly rectangle building with mismatched windows | ![](random_w11_collage.png) | ![](random_w11_doodle.png) | ![](random_w11_pixel.png) |
| 22 | Week 2026-04-27 ~ 2026-05-03 \| Total water: 3600 ml across 16 sips \| Avg/day: 514.3 ml \| Dominant weather: cloudy (avg 7.1 C) \| Place categories: {'restaurant': 2, 'landmark': 2, 'work': 2, 'market': 2, 'home': 3, 'cafe': 1} \| Top places: Apartment, Restaurant, Tower, Workplace, Market | 8 mismatched water cups arranged in a row; lumpy clouds doodled across the top; bowl with wavy noodles; tower drawn with shaky vertical lines; ugly rectangle building with mismatched windows; shopping cart with random shapes inside | ![](random_w22_collage.png) | ![](random_w22_doodle.png) | ![](random_w22_pixel.png) |
| 33 | Week 2026-04-27 ~ 2026-05-03 \| Total water: 3250 ml across 13 sips \| Avg/day: 464.3 ml \| Dominant weather: snowy (avg 14.7 C) \| Place categories: {'landmark': 1, 'market': 3, 'bookstore': 2, 'work': 1} \| Top places: Market, Bookstore, Tower, Workplace | 8 mismatched water cups arranged in a row; scattered snowflakes drawn as asterisks; tower drawn with shaky vertical lines; shopping cart with random shapes inside; stack of crooked books; ugly rectangle building with mismatched windows | ![](random_w33_collage.png) | ![](random_w33_doodle.png) | ![](random_w33_pixel.png) |
| 44 | Week 2026-04-27 ~ 2026-05-03 \| Total water: 3200 ml across 13 sips \| Avg/day: 457.1 ml \| Dominant weather: windy (avg 9.6 C) \| Place categories: {'market': 2, 'restaurant': 3, 'work': 2, 'home': 1, 'cafe': 1, 'landmark': 1} \| Top places: Restaurant, Market, Workplace, Apartment, Cafe | 8 mismatched water cups arranged in a row; squiggly wind lines, leaves flying sideways; shopping cart with random shapes inside; bowl with wavy noodles; ugly rectangle building with mismatched windows; tiny house with a triangle roof | ![](random_w44_collage.png) | ![](random_w44_doodle.png) | ![](random_w44_pixel.png) |
