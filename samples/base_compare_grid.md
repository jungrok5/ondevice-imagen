# Commercial-friendly base × style LoRA shootout

Same SDXL 1.0 base (Open RAIL++ M, commercial OK). Two speed
regimes — full 30-step base inference vs SDXL Lightning
(loaded as a LoRA over the same base, 4 step). Two style LoRAs
stacked on top of each.

Input event: 19:00 / 맑음 / 2026-12-25 / 무궁화 아파트  
Data phrases: `evening dusk, lamps lit, winter, bare branches, Christmas day, fairy lights, festive, clear sky, Korean setting, tall apartment buildings, forested area, tall pines`

| base regime | style LoRA | result |
| --- | --- | --- |
| `base_30step` (30 step) | `worstimever` | ![](base_compare_base_30step_worstimever.png) |
| `base_30step` (30 step) | `mspaint_portraits` | ![](base_compare_base_30step_mspaint_portraits.png) |
| `lightning_4step` (4 step) | `worstimever` | ![](base_compare_lightning_4step_worstimever.png) |
| `lightning_4step` (4 step) | `mspaint_portraits` | ![](base_compare_lightning_4step_mspaint_portraits.png) |
