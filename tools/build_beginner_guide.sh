#!/usr/bin/env bash
# Requires Python with matplotlib/scipy, pandoc and XeLaTeX plus DejaVu fonts/fvextra/fancyhdr.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/strain2d-guide-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/strain2d-guide-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python tools/make_guide_figures.py
pandoc docs/BEGINNER_GUIDE.md --standalone \
  --resource-path=docs --pdf-engine=xelatex \
  --toc --number-sections --highlight-style=tango \
  --output docs/Strain2D_Beginners_Handbook.pdf
