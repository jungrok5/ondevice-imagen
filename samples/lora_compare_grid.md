# LoRA shootout — same event, 4 LoRAs

Input event: 19:00 / 맑음 / 2026-12-25 / 대한민국 서울시 무궁화 아파트

Data phrases injected: `evening dusk, lamps lit, winter, bare branches, Christmas day, fairy lights, festive, clear sky, Korean setting, tall apartment buildings, forested area, tall pines`

Same SDXL-Turbo base, same prompt structure, same seed (42). The
only thing changing per row is the loaded LoRA + its trigger.

| LoRA | trigger | result |
| --- | --- | --- |
| `worstimever` | `DD-wte artstyle, worst-im-ever cartoon doodle` | ![](lora_compare_worstimever.png) |
| `mspaint_portraits` | `MSPaint drawing` | ![](lora_compare_mspaint_portraits.png) |
| `lah_cute_social` | `cute doodle` | ![](lora_compare_lah_cute_social.png) |
| `pixel_art_xl` | `(no trigger)` | ![](lora_compare_pixel_art_xl.png) |
