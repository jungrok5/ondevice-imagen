# Raw LoRA capability — trigger + subject only

No STYLE_TAGS prefix, no NEGATIVE prompt, no translated phrase
blocks. Just `{LoRA trigger}, {subject}`. Same subject across
all rows; same seed (42); base SDXL-Turbo.

Subject: `a young man holding a coffee cup at a cafe table`

| LoRA | trigger | full prompt | result |
| --- | --- | --- | --- |
| `00_baseline_no_lora` | `(none)` | `a young man holding a coffee cup at a cafe table` | ![](raw_lora_00_baseline_no_lora.png) |
| `01_worstimever` | `DD-wte artstyle` | `DD-wte artstyle, a young man holding a coffee cup at a cafe table` | ![](raw_lora_01_worstimever.png) |
| `02_mspaint_portraits` | `MSPaint drawing of` | `MSPaint drawing of a young man holding a coffee cup at a cafe table` | ![](raw_lora_02_mspaint_portraits.png) |
| `03_pixel_art_xl` | `pixel art` | `pixel art, a young man holding a coffee cup at a cafe table` | ![](raw_lora_03_pixel_art_xl.png) |
| `04_lah_cute_social` | `cute doodle` | `cute doodle, a young man holding a coffee cup at a cafe table` | ![](raw_lora_04_lah_cute_social.png) |
