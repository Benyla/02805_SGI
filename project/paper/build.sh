#!/usr/bin/env bash
# Build the hand-in PDF: paper/report.pdf = front page + 6-page PNAS paper.
# Needs tectonic (brew install tectonic) and the outputs of the analysis scripts.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python"
( cd .. && $PY -m src.paper_numbers )
[ -f cover_mosaic.jpg ] || ( cd .. && $PY -m src.figures.cover_mosaic )
[ -f DTULogo.png ] || cp "DTULogo (1).png" DTULogo.png
tectonic frontpage.tex
tectonic main.tex
tectonic report.tex
echo "built paper/report.pdf"
