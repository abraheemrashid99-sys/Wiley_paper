#!/usr/bin/env bash
# Builds the submission package: dist/DNSC_IJCS_manuscript.pdf and
# dist/DNSC_IJCS_manuscript.zip (LaTeX sources, figures, diagrams), compiled
# from a clean copy so that the archive is known to build on its own.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
paper="$root/paper"; dist="$root/dist"; name="DNSC_IJCS_manuscript"
stage="$(mktemp -d)"; pkg="$stage/$name"
mkdir -p "$pkg/figures" "$pkg/diagrams" "$dist"
cp "$paper"/{main,preamble,frontmatter,endmatter,references}.tex "$paper"/sec*.tex "$pkg/"
cp "$paper"/{README.md,REFERENCES_CHECK.md,NUMBERS.md,numbers.json} "$pkg/"
cp "$paper"/figures/*.pdf "$pkg/figures/"
cp "$paper"/diagrams/bw_style.tex "$paper"/diagrams/fig_*.tex "$paper"/diagrams/fig_*.pdf "$pkg/diagrams/"
( cd "$pkg/diagrams" && for f in fig_*.tex; do pdflatex -interaction=nonstopmode "$f" >/dev/null; done; rm -f *.aux *.log )
( cd "$pkg" && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null )
if grep -q "Overfull\|undefined" "$pkg/main.log"; then echo "warnings in main.log" >&2; grep "Overfull\|undefined" "$pkg/main.log" >&2; fi
cp "$pkg/main.pdf" "$dist/$name.pdf"
mv "$pkg/main.pdf" "$pkg/$name.pdf"
rm -f "$pkg"/*.aux "$pkg"/*.log "$pkg"/*.out
rm -f "$dist/$name.zip"
( cd "$stage" && zip -qr "$dist/$name.zip" "$name" )
rm -rf "$stage"
echo "built $dist/$name.pdf and $dist/$name.zip"
