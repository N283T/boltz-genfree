#!/usr/bin/env bash
# Assemble the public site into a directory (default: _site). CI and local preview use this same script.
#   docs/site/assemble.sh [outdir]      then:  python3 -m http.server -d _site 8000
# Layout of the result:
#   index.html  research.html  site.css  logo-*.svg  assets/        the site pages and shared files
#   docs/*.html                                                      Docs, built from README.md and docs/*.md
#   webgpu/web/screen.html  webgpu/src/  webgpu/packs/               the browser tool and what it loads
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${1:-_site}"
rm -rf "$out"
mkdir -p "$out/docs" "$out/webgpu/web" "$out/webgpu/packs"

cp docs/site/index.html docs/site/research.html docs/site/site.css docs/site/logo-on-dark.svg docs/site/logo-on-light.svg "$out/"
cp -r docs/site/assets "$out/"

# Docs pages (docs/site/build_docs.py writes every page into the directory it is given)
uv run docs/site/build_docs.py "$out/docs"

# The browser tool: its sources and the affinity pack it reads (the trunk weights come from Hugging Face)
cp -r webgpu/src "$out/webgpu/"
cp webgpu/web/screen.html "$out/webgpu/web/"
if [ -f webgpu/packs/genfree-affinity.json ]; then
  cp webgpu/packs/genfree-affinity.json webgpu/packs/genfree-affinity.bin "$out/webgpu/packs/"
else
  echo "note: webgpu/packs/genfree-affinity.* not found; the tool will not run until the affinity pack is added" >&2
fi
echo "site assembled in $out"
