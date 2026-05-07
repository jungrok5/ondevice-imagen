"""Compare our kasun output to the user's ChatGPT-tested ground truth.

Reads pairs from samples/reference/user_test/{line,color}/ — files named
set<N>_input.* and set<N>_output.* — and produces a side-by-side grid:

  | original input | ChatGPT (ground truth) | our kasun output | (delta) |

So we can tune the filter parameters per mode against real references
instead of guessing.

Usage:
  py scripts/compare_user_test.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kasun_filter import kasun_line, kasun_color  # noqa: E402

USER_TEST = ROOT / "samples" / "reference" / "user_test"
OUT_DIR = ROOT / "samples"


SET_RE = re.compile(r"^set(\d+)_input\.")


def find_sets(folder: Path) -> list[tuple[str, Path, Path]]:
    """Return [(set_id, input_path, output_path), ...] in folder."""
    if not folder.exists():
        return []
    pairs: list[tuple[str, Path, Path]] = []
    for p in sorted(folder.iterdir()):
        m = SET_RE.match(p.name)
        if not m:
            continue
        set_id = m.group(1)
        # match output regardless of extension
        outputs = list(folder.glob(f"set{set_id}_output.*"))
        if outputs:
            pairs.append((set_id, p, outputs[0]))
    return pairs


def run(mode: str) -> list[dict]:
    folder = USER_TEST / mode
    sets = find_sets(folder)
    if not sets:
        print(f"[compare] no sets found in {folder} -- skipping {mode}")
        return []

    rows = []
    fn = kasun_line if mode == "line" else kasun_color
    for set_id, in_path, gt_path in sets:
        img = Image.open(in_path)
        ours = fn(img)
        ours_path = OUT_DIR / f"compare_{mode}_set{set_id}_ours.png"
        ours.save(ours_path)

        rows.append({
            "set": set_id,
            "input": in_path.relative_to(OUT_DIR),
            "gt": gt_path.relative_to(OUT_DIR),
            "ours": ours_path.relative_to(OUT_DIR),
        })
        print(f"[{mode}] set{set_id}: {in_path.name} → {ours_path.name}")
    return rows


def write_index(line_rows: list[dict], color_rows: list[dict]) -> None:
    lines = [
        "# kasun vs ChatGPT ground truth",
        "",
        "User-tested input/output pairs from ChatGPT's official 낙서풍",
        "template, run through our kasun_line / kasun_color filters.",
        "Side-by-side so we can see exactly where we still diverge.",
        "",
    ]
    if line_rows:
        lines += ["## 단색선 (line)", "", "| set | input | ChatGPT | ours |", "| --- | --- | --- | --- |"]
        for r in line_rows:
            lines.append(
                f"| {r['set']} | ![]({r['input'].as_posix()}) "
                f"| ![]({r['gt'].as_posix()}) "
                f"| ![]({r['ours'].as_posix()}) |"
            )
        lines.append("")
    if color_rows:
        lines += ["## 컬러 낙서풍 (color)", "", "| set | input | ChatGPT | ours |", "| --- | --- | --- | --- |"]
        for r in color_rows:
            lines.append(
                f"| {r['set']} | ![]({r['input'].as_posix()}) "
                f"| ![]({r['gt'].as_posix()}) "
                f"| ![]({r['ours'].as_posix()}) |"
            )
        lines.append("")
    if not line_rows and not color_rows:
        lines += [
            "No reference sets dropped yet. Add files at",
            "`samples/reference/user_test/line/setN_input.*` and",
            "`samples/reference/user_test/line/setN_output.*` (and same",
            "under `color/`), then re-run this script.",
        ]
    (OUT_DIR / "compare_user_test.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\n[compare] index → samples/compare_user_test.md")


def main() -> None:
    line_rows = run("line")
    color_rows = run("color")
    write_index(line_rows, color_rows)


if __name__ == "__main__":
    main()
