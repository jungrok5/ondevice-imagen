# LoRA fuse/unload sanity check

Same SDXL-Turbo, same data prompt, same seed. Only the LoRA
state and trigger phrase change row-to-row.

| label | LoRA state | prompt | result |
| --- | --- | --- | --- |
| A | none | data only | ![](sanity_A_no_lora_base_prompt.png) |
| B | worstimever fused @ 0.9 | trigger + data | ![](sanity_B_worst_fused_with_trigger.png) |
| C | worstimever still fused | data only (NO trigger) | ![](sanity_C_worst_fused_no_trigger.png) |
| D | after unfuse + unload | data only | ![](sanity_D_after_unload_base_prompt.png) |

Expected:
  - A and D should match (LoRA fully unloaded by D)
  - B should differ noticeably from A/D (trigger activates style)
  - C tells us how much LoRA does WITHOUT trigger — small or large?
