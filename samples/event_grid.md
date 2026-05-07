# Single-event prompt: 2026-12-25 19:00 무궁화

## Input event

| field | value |
| --- | --- |
| time | 19:00 |
| weather | 맑음 |
| date | 2026-12-25 |
| country | 대한민국 |
| city | 서울시 |
| place | 무궁화 아파트 |

## Field → phrase mapping (from src/event_prompt.py)

| source field | phrase injected |
| --- | --- |
| `time=19:00` | `evening dusk, lamps lit` |
| `date=2026-12-25 (season)` | `winter, bare branches` |
| `date=2026-12-25 (special)` | `Christmas day, fairy lights, festive` |
| `weather=맑음` | `clear sky` |
| `country=대한민국` | `Korean setting` |
| `place='무궁화 아파트' matches '아파트'` | `tall apartment buildings` |
| `place='무궁화 아파트' matches '포레스트'` | `forested area, tall pines` |

## Full positive prompt (sent to SDXL-Turbo + worstimever LoRA)

```
DD-wte artstyle, worst-im-ever cartoon doodle, ugly MS Paint doodle, white paper, black ink only, pixelated low-res, child scribble, naive crude drawing, evening dusk, lamps lit, winter, bare branches, Christmas day, fairy lights, festive, clear sky, Korean setting, tall apartment buildings, forested area, tall pines
```

## Negative prompt

```
photorealistic, sharp focus, polished, professional, hd, anti-aliased, smooth gradient, oil painting
```

## Result

![](event_xmas_19h.png)
