# Trigger correctness — old (guess) vs new (Civitai-official)

Same `EVENT`, same `prompt_seed=100`, same `DIFFUSION_SEED=42`.
Only the trigger phrase changes between cells in a row.

| LoRA | OLD trigger | NEW trigger |
| --- | --- | --- |
| **worstimever** | ![](trig_worstimever_OLD.png)<br>`DD-wte artstyle, worst-im-ever cartoon doodle` | ![](trig_worstimever_NEW.png)<br>`WTE artstyle` |
| **mspaint_portraits** | ![](trig_mspaint_portraits_OLD.png)<br>`MSPaint drawing of` | ![](trig_mspaint_portraits_NEW.png)<br>`MSPaint portrait` |
