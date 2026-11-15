#!/bin/zsh
# Renders docs/figures/*.pptx to docs/figures/png/*.png (Keynote -> PDF -> 3000-px PNG via Quick Look).
# These PNGs (Figures 2 and 4) contain patient images: they are embedded in the manuscript for the author.
# Usage: zsh docs/build/render_figures.sh [subdir]   (subdir "_layout_check" renders the placeholder copies)
set -e
ROOT=${0:A:h:h:h}
SRC="$ROOT/docs/figures/${1:-}"
OUTD="$SRC/png"; mkdir -p "$OUTD"
K=~/Library/Containers/com.apple.iWork.Keynote/Data/figrender; mkdir -p "$K"
for f in "$SRC"/*.pptx; do
  b=${f:t:r}
  osascript -e "tell application \"Keynote\"
    set d to open POSIX file \"$f\"
    delay 2
    export d to POSIX file \"$K/$b.pdf\" as PDF
    close d saving no
  end tell" >/dev/null
  qlmanage -t -s 3000 -o "$OUTD" "$K/$b.pdf" >/dev/null 2>&1
  mv -f "$OUTD/$b.pdf.png" "$OUTD/$b.png"
  "$ROOT/.venv/bin/python" -W ignore "$ROOT/docs/build/trim_png.py" "$OUTD/$b.png" 2>/dev/null
done
ls "$OUTD"
