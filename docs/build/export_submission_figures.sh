#!/bin/zsh
# Submission figure files for JCMR: one TIFF per figure (LZW, 600 dpi at its PowerPoint width), numbered in order of citation.
# Source: the editable PowerPoint files in docs/figures/ (real images).
# Output: docs/submission/figures/{Figure_1..12, Figure_S1, Graphical_abstract}.tif (main figures in order of first citation)
set -e
ROOT=${0:A:h:h:h}; FIG="$ROOT/docs/figures"; OUTD="$ROOT/docs/submission/figures"; mkdir -p "$OUTD"
K=~/Library/Containers/com.apple.iWork.Keynote/Data/figrender_submission; mkdir -p "$K"
typeset -A MAP
MAP=(Figure_1_cohort Figure_1 Figure_2_design Figure_2 Figure_7_one_exam Figure_3 Figure_3_retrieval Figure_4 Figure_4_examples Figure_5
     Figure_8_scaling Figure_6 Figure_9_scorecard Figure_7 Figure_10_external Figure_8 Figure_15_search Figure_9 Figure_12_risk Figure_10
     Figure_11_shortcut Figure_11 Figure_13_saliency Figure_12 Figure_14_head_to_head Figure_S1 Graphical_abstract Graphical_abstract)
for src dst in ${(kv)MAP}; do
  f="$FIG/$src.pptx"; [[ -f "$f" ]] || { echo "missing $src"; continue; }
  osascript -e "tell application \"Keynote\"
    set d to open POSIX file \"$f\"
    delay 2
    export d to POSIX file \"$K/$dst.pdf\" as PDF
    close d saving no
  end tell" >/dev/null
  qlmanage -t -s 4500 -o "$K" "$K/$dst.pdf" >/dev/null 2>&1
  "$ROOT/.venv/bin/python" -W ignore -c "
import subprocess, sys; subprocess.run([sys.executable, '-W', 'ignore', '$ROOT/docs/build/trim_png.py', '$K/$dst.pdf.png'])
from PIL import Image; im = Image.open('$K/$dst.pdf.png').convert('RGB'); im.save('$OUTD/$dst.tif', compression='tiff_lzw', dpi=(600, 600)); print('$dst', im.size)" 2>/dev/null
done
