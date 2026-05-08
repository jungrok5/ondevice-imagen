"""Download the 14 candidate LoRAs from Civitai for the catalog test.

The two LoRAs already in models/lora/ (worstimever_xl, sdxl_mspaint_portraits)
are not re-downloaded.

Civitai returns either the file directly (anonymous OK) or a 401/redirect
to a login page when the model requires consenting to T&Cs first. For
flagged models the user has to set CIVITAI_TOKEN. We try anonymous and
flag failures so the user can fill in the token only for what failed.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "models" / "lora"
LORA_DIR.mkdir(parents=True, exist_ok=True)

# (target_filename, civitai_versionId, model_id_for_logs)
TARGETS = [
    ("doodle_style.safetensors",                733335,   820070),
    ("doodles_in_real_life.safetensors",        229000,   258449),
    ("cyberpunk_lines.safetensors",             574889,   640918),
    ("linedrawing.safetensors",                 709714,   793831),
    ("lineart_zoolin.safetensors",              2148315,  2429855),
    ("asian_line_storyboard.safetensors",       429761,   478811),
    ("soft_squishy_linework.safetensors",       515703,   573071),
    ("colored_line.safetensors",                598326,   668536),
    ("clean_bw_line_art.safetensors",           1010573,  1132776),
    ("tangbohu_landscape.safetensors",          535374,   595127),
    ("digital_art_illustrations.safetensors",   534574,   594206),
    ("simplex.safetensors",                     300268,   337229),
    ("simple_toons.safetensors",                2007180,  2271820),
    ("jackledead_artstyle.safetensors",         783895,   876641),
    ("japanese_illustration.safetensors",       1841264,  2083690),
]

# Token resolution: env var first, then a local .civitai_token file
# (gitignored). The file fallback exists because Claude Code inherits
# the parent process env and won't see env vars the user set in a new
# shell session — a local file is the simplest workaround that doesn't
# require restarting Claude Code.
TOKEN = os.environ.get("CIVITAI_TOKEN", "")
if not TOKEN:
    token_file = ROOT / ".civitai_token"
    if token_file.exists():
        TOKEN = token_file.read_text(encoding="utf-8").strip()


def download(target: Path, version_id: int, model_id: int) -> tuple[bool, str]:
    if target.exists() and target.stat().st_size > 1_000_000:
        return True, f"already present ({target.stat().st_size // 1_000_000} MB)"

    url = f"https://civitai.com/api/download/models/{version_id}"
    if TOKEN:
        url += f"?token={TOKEN}"

    req = urllib.request.Request(url, headers={"User-Agent": "local-ai-rnd/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "html" in ctype:
                # Got the login page instead of the safetensor binary
                return False, f"got HTML (auth wall) — needs CIVITAI_TOKEN"
            total = int(resp.headers.get("Content-Length", "0"))
            tmp = target.with_suffix(".part")
            t = time.time()
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(target)
            mb = target.stat().st_size // 1_000_000
            return True, f"{mb} MB in {time.time() - t:.1f}s"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    print(f"[dl] CIVITAI_TOKEN: {'set' if TOKEN else 'not set'}")
    print(f"[dl] target dir: {LORA_DIR}")
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for fname, model_id, version_id in TARGETS:
        target = LORA_DIR / fname
        print(f"\n[dl] {fname} (model {model_id}, version {version_id})")
        ok, msg = download(target, version_id, model_id)
        marker = "[OK]" if ok else "[FAIL]"
        print(f"[dl]   {marker} {msg}")
        if ok:
            successes.append(fname)
        else:
            failures.append((fname, msg))

    print("\n" + "=" * 60)
    print(f"[dl] {len(successes)} ok, {len(failures)} failed")
    if failures:
        print("[dl] failures:")
        for fname, msg in failures:
            print(f"[dl]   - {fname}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
