#!/bin/sh
# Export docs/orgmode to docs/source as reStructuredText.
# Exported rst is not committed. Run from anywhere.

set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
src="$root/docs/orgmode"
dst="$root/docs/source"

find "$src" -type f -name '*.org' | sort | while IFS= read -r org; do
  rel=${org#"$src/"}
  out="$dst/${rel%.org}.rst"
  mkdir -p "$(dirname "$out")"
  pandoc -f org -t rst -o "$out" "$org"
done
