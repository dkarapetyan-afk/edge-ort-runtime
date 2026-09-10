#!/usr/bin/env bash
# Extract Noto Sans CJK SC/TC/JP/KR TTFs for egui (from system .ttc).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/assets/fonts"
mkdir -p "$OUT"

TTC="${NOTO_CJK_TTC:-}"
if [[ -z "$TTC" ]]; then
  for c in \
    /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
    /usr/share/fonts/opentype/noto/NotoSansCJK.ttc
  do
    if [[ -f "$c" ]]; then TTC="$c"; break; fi
  done
fi
if [[ -z "${TTC:-}" || ! -f "$TTC" ]]; then
  echo "NotoSansCJK-Regular.ttc not found. Install fonts-noto-cjk or set NOTO_CJK_TTC=."
  exit 1
fi

if ! python3 -c "import fontTools" 2>/dev/null; then
  VENV="${CJK_FONT_VENV:-/tmp/edge-ort-fonttools-venv}"
  if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q fonttools
  fi
  PY="$VENV/bin/python"
else
  PY=python3
fi

"$PY" - "$TTC" "$OUT" <<'PY'
import sys
from pathlib import Path
from fontTools.ttLib import TTCollection

ttc_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
want = {
    "Noto Sans CJK SC": "NotoSansCJKsc-Regular.ttf",
    "Noto Sans CJK TC": "NotoSansCJKtc-Regular.ttf",
    "Noto Sans CJK JP": "NotoSansCJKjp-Regular.ttf",
    "Noto Sans CJK KR": "NotoSansCJKkr-Regular.ttf",
}
col = TTCollection(str(ttc_path))
saved = []
for font in col.fonts:
    names = []
    for rec in font["name"].names:
        if rec.nameID in (1, 4, 16):
            try:
                names.append(rec.toUnicode())
            except Exception:
                pass
    joined = " | ".join(names)
    for key, outname in want.items():
        if key in joined:
            out = out_dir / outname
            if not out.exists():
                font.save(str(out))
                print(f"[ok] {outname} ({out.stat().st_size} bytes)")
            else:
                print(f"[skip] {outname} already present")
            saved.append(outname)
print(f"done: {len(saved)} faces from {ttc_path}")
PY
