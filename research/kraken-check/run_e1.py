#!/usr/bin/env python3
"""E1: KRAKEN inside FishLab's arbiter. Sequential blocks of 100 deals x 6 rotations,
each block its own seed, one JSON row per block appended to e1_results.jsonl (resumable)."""
import json, subprocess, sys, time, os
A = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(A, "..", "..", "engine"))
OUT = os.path.join(A, "e1_results.jsonl"); LOG = os.path.join(A, "e1_progress.log")
SPEC = json.load(open(os.path.join(ENGINE, "fishbot_v07.json")))["spec"]
SEARCH = 's1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26'
FCHEAP = 'v06:' + SEARCH
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
cells = [
  ("default",          [],                                           "kraken", SPEC,     777001, 9),
  ("nooot",            ["--no-out-of-turn"],                         "kraken", SPEC,     778001, 10),
  ("mirror_kraken",    [],                                           "kraken", "kraken", 781001, 2),
  ("kv6_vs_v07",       [],                                           "kv6",    SPEC,     782001, 5),
  ("kraken_vs_v06",    [],                                           "kraken", "v06",    783001, 5),
  ("nooot_nocardless", ["--no-out-of-turn", "--no-cardless-declare"], "kraken", SPEC,     779001, 5),
  ("legacy",           ["--legacy"],                                 "kraken", SPEC,     780001, 5),
  ("kraken_vs_fcheap", [],                                           "kraken", FCHEAP,   784001, 3),
  ("kraken_vs_v05",    [],                                           "kraken", "v05",    785001, 3),
  ("kraken_vs_v04",    [],                                           "kraken", "v04",    786001, 3),
  # KV FishBot v0.6 on the SAME seeds as the default KRAKEN cell, so the two are paired at block level
  ("kv6_vs_v07_paired", [],                                          "kv6",    SPEC,     777001, 5),
]
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        if line.strip():
            r = json.loads(line); done.add((r["cell"], r["seed"]))
def log(msg):
    with open(LOG, "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")
log(f"start threads={THREADS}")
for cell, flags, a, b, seed0, nblocks in cells:
    for k in range(nblocks):
        seed = seed0 + k
        if (cell, seed) in done: continue
        cmd = [os.path.join(ENGINE, "fish"), "match", f"--a={a}", f"--b={b}", "--games=100",
               "--rotations=6", f"--threads={THREADS}", f"--seed={seed}", "--json"] + flags
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True)
        rowline = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode != 0 or not rowline:
            log(f"FAIL {cell} seed={seed} rc={p.returncode} stderr={p.stderr[-600:]}"); continue
        row = json.loads(rowline[-1])
        row.update({"cell": cell, "flags": flags, "seed": seed, "wall": round(time.time() - t0, 1), "argv": cmd})
        with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
        log(f"{cell} seed={seed} winA={row['winRateA']:.4f} setsA={row['meanSetsA']:.3f} setsB={row['meanSetsB']:.3f} "
            f"declA={row['declAccA']:.3f} declB={row['declAccB']:.3f} {row['wall']}s")
log("ALL DONE")
