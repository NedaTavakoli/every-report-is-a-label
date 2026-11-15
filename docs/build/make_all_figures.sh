#!/bin/zsh
# Rebuilds every paper figure from the result files in outputs/ (written by the analysis scripts in src/).
#   zsh docs/build/make_all_figures.sh            editable PowerPoint figures + PNG renders + 600-dpi TIFFs
#   zsh docs/build/make_all_figures.sh --images   first extract the example patient images (needs the DICOM data locally;
#                                                 set FULL_EXAM_DICOM_ROOT for the whole-examination figure)
# Rendering uses Keynote + Quick Look (macOS). FIG_TEXT_SCALE sets text size (default 1.25); FIG_PLACEHOLDER=1 builds
# layout-check copies with gray boxes instead of patient images.
set -e
ROOT=${0:A:h:h:h}; PY=("$ROOT/.venv/bin/python" -W ignore); [[ -x "$ROOT/.venv/bin/python" ]] || PY=(python3 -W ignore); B="$ROOT/docs/build"
cd "$B"
if [[ "$1" == "--images" ]]; then
  for s in extract_example_slices.py extract_example_reports.py extract_whole_exam.py extract_fig7_report.py extract_search_images.py; do $PY $s || echo "($s failed)"; done
fi
$PY make_figures.py >/dev/null; $PY make_figures_v05.py
FIG_PLACEHOLDER=1 $PY make_figures.py >/dev/null; FIG_PLACEHOLDER=1 $PY make_figures_v05.py >/dev/null
zsh "$B/render_figures.sh" >/dev/null; zsh "$B/render_figures.sh" _layout_check >/dev/null
zsh "$B/export_submission_figures.sh"
