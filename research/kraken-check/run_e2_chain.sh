#!/bin/sh
# The KV-arbiter runs, in order.  Identical deal seeds (910000+) across variants make them paired.
A="/Users/dylan/Documents/GitHub/fish optimization/research/kraken-check"; W="/Users/dylan/Documents/GitHub/fish optimization/engine/external/kvarena-work"; PY="/Users/dylan/Documents/GitHub/fish optimization/engine/external/kv-venv/bin/python"
cd "$W"
for v in rev2 dealthand ladder seedvary dealtseed; do
  echo "=== variant $v start $(date +%H:%M:%S)"
  E2_VARIANT=$v "$PY" run_e2.py 1000 6 910000 "$A/e2_$v.jsonl" 2>&1 | tail -40
done
echo "=== big corrected run start $(date +%H:%M:%S)"
E2_VARIANT=dealthand "$PY" run_e2.py 4000 10 920000 "$A/e2_dealthand_big.jsonl" 2>&1 | tail -40
echo "=== E2 chain done $(date +%H:%M:%S)"
