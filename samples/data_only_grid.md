# Data-only baseline — just translated event, no styling

User wanted to see: 'what does SDXL produce if I just give it
my translated event data, no STYLE_TAGS, no LoRA trigger phrase,
no NEGATIVE prompt?'

Input event: 19:00 / 맑음 / 2026-12-25 / 대한민국 서울시 무궁화 아파트  
Data-only prompt: `evening dusk, lamps lit, winter, bare branches, Christmas day, fairy lights, festive, clear sky, Korean setting, tall apartment buildings`

| LoRA fused | result |
| --- | --- |
| `(no LoRA)` | ![](data_only_00_data_only_no_lora.png) |
| `worstimever_xl.safetensors` | ![](data_only_01_data_only_worstimever.png) |
| `sdxl_mspaint_portraits.safetensors` | ![](data_only_02_data_only_mspaint.png) |
