# Samples

Reference images committed to the repo so they can be viewed on GitHub
without running the prototype locally.

- `base_collage.png` — the rule-rendered collage from `data/sample_week.json`.
  This is the deterministic input for img2img.
- `sd_img2img.png` — img2img output (collage redrawn in the bad-doodle style).
- `sd_txt2img.png` — txt2img output for comparison (no collage seed).

Local generations land in `outputs/` (gitignored). Re-generate the samples here with:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_samples.py
```
