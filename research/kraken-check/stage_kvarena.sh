#!/bin/sh
# Stage KV's arbiter for the SESTINA-inside-their-engine runs.  Default work dir is
# engine/external/kvarena-work (gitignored, like the vendored package it copies).
#   sh research/kraken-check/stage_kvarena.sh [workdir]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WORK=${1:-"$REPO/engine/external/kvarena-work"}
rm -rf "$WORK" && mkdir -p "$WORK/external_v07"
cp -R "$REPO/engine/external/kraken/fish" "$REPO/engine/external/kraken/fish4" "$REPO/engine/external/kraken/kraken" "$WORK/"
find "$WORK" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$REPO/engine/external/kv-fishbot-v06/external_v07/shim_decide.cpp" "$REPO/engine/external/kv-fishbot-v06/external_v07/v07_spec.txt" "$WORK/external_v07/"
cp "$HERE/kvarena/shim_decide2.cpp" "$WORK/external_v07/"
cp "$HERE/kvarena/run_e2.py" "$HERE/kvarena/e2_variants.py" "$WORK/"
FLAGS="-std=c++20 -O3 -march=native -funroll-loops -fno-math-errno -w -pthread"
clang++ $FLAGS -I "$REPO/engine/src" -o "$WORK/external_v07/fish_v07_decide"  "$WORK/external_v07/shim_decide.cpp"
clang++ $FLAGS -I "$REPO/engine/src" -o "$WORK/external_v07/fish_v07_decide2" "$WORK/external_v07/shim_decide2.cpp"
python3 -c "import json,sys;a=json.load(open('$REPO/engine/fishbot_v07.json'))['allparamsSpec'];b=open('$WORK/external_v07/v07_spec.txt').read().strip();sys.exit(0 if a==b else 1)" || { echo "v07_spec.txt does not match fishbot_v07.json"; exit 1; }
echo "staged in $WORK"
