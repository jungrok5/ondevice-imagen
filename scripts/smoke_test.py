"""Quick non-GPU smoke test for the pure-Python pieces."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import diary, prompt_builder, base_collage  # noqa: E402

week = diary.load_week(ROOT / "data" / "sample_week.json")
summary = diary.summarize(week)
print("SUMMARY:", diary.summary_to_text(summary))

built = prompt_builder.build(summary)
print()
print("POSITIVE:", built.positive)
print()
print("NEGATIVE:", built.negative)
print()
print("DEBUG:", built.debug_summary)

img = base_collage.render(summary)
out = ROOT / "outputs" / "smoke_collage.png"
out.parent.mkdir(exist_ok=True)
img.save(out)
print()
print(f"BASE COLLAGE saved to: {out}")
