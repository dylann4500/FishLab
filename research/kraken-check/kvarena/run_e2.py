#!/usr/bin/env python3
"""E2/E3: KRAKEN's headline design inside KV's OWN arbiter (their engine, their rules -- claims on
turn only, misdeclarations awarded to the opponents -- their bridge), on fresh seeds, resumable.
E2_VARIANT selects the bridge: rev2 (theirs as published) or a one-change variant (e2_variants.py)."""
import json, os, sys, time, multiprocessing as mp
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from fish.cards import team_of
from fish.engine import AskEvent, ClaimEvent, PassEvent, GameState
from fish.observation import Observation
from fish.rules import RuleConfig
from fish4.registry4 import V06_DEPLOYED, make_agent

RULES = RuleConfig(wrong_distribution_outcome="opponent")
AGENT0 = 9100
VARIANT = os.environ.get("E2_VARIANT", "rev2")

def make_dylan():
    if VARIANT == "rev2":
        return make_agent(("dylan_v07", {}))
    import e2_variants
    return e2_variants.make(VARIANT)

def play(task):
    deal_seed, kv_even = task
    agents = []
    for p in range(6):
        kv = (p % 2 == 0) == kv_even
        agents.append(make_agent(V06_DEPLOYED) if kv else make_dylan())
    st = GameState.deal(RULES, seed=deal_seed)
    for p, a in enumerate(agents):
        a.begin_game(p, RULES, AGENT0 + deal_seed * 13 + p)
    t0 = time.time()
    forced = {"kv": [0, 0], "dy": [0, 0]}      # [total, correct]
    plies = 0
    for _ in range(600):
        if st.is_terminal: break
        mover = st.turn
        was_forced = st.must_claim(mover)
        obs = Observation.from_state(st, mover)
        act = agents[mover].act(obs)
        ev = st.apply(mover, act)
        plies += 1
        if was_forced and isinstance(ev, ClaimEvent):
            side = "kv" if ((mover % 2 == 0) == kv_even) else "dy"
            forced[side][0] += 1
            forced[side][1] += int(ev.winner == team_of(ev.claimer))
    kv_team = 0 if kv_even else 1
    kv = sum(1 for w in st.set_winner if w == kv_team)
    dy = sum(1 for w in st.set_winner if w == 1 - kv_team)
    asks = {"kv": [0, 0], "dy": [0, 0]}     # [hits, total]
    decl = {"kv": [0, 0], "dy": [0, 0]}     # [correct, total]
    own_err = {"kv": 0, "dy": 0}; alloc_err = {"kv": 0, "dy": 0}
    for ev in st.history:
        if isinstance(ev, AskEvent):
            side = "kv" if team_of(ev.asker) == kv_team else "dy"
            asks[side][1] += 1; asks[side][0] += int(ev.success)
        elif isinstance(ev, ClaimEvent):
            side = "kv" if team_of(ev.claimer) == kv_team else "dy"
            team = team_of(ev.claimer)
            ok = ev.winner == team
            decl[side][1] += 1; decl[side][0] += int(ok)
            if not ok:
                if any(team_of(h) != team for h in ev.revealed): own_err[side] += 1
                else: alloc_err[side] += 1
    dylans = [a for a in agents if getattr(a, "name", "") == "dylan_v07"]
    extra = {}
    for a in dylans:
        for k, v in getattr(a, "stats", {}).items(): extra[k] = extra.get(k, 0) + v
    return {"deal": deal_seed, "kv_even": kv_even, "variant": VARIANT, "kv": kv, "dylan": dy,
            "margin": kv - dy, "terminal": st.is_terminal, "plies": plies,
            "fallbacks": sum(getattr(a, "fallbacks", 0) for a in agents),
            "off_team_owners": sum(getattr(a, "off_team_owners", 0) for a in agents),
            "asks": asks, "decl": decl, "forced": forced, "own_err": own_err, "alloc_err": alloc_err,
            "extra": extra, "seconds": round(time.time() - t0, 2)}

def summarize(rows):
    n = len(rows)
    m = [r["margin"] for r in rows]; mean = sum(m) / n
    var = sum((x - mean) ** 2 for x in m) / max(1, n - 1); se = (var / n) ** 0.5
    kv_sets = sum(r["kv"] for r in rows); dy_sets = sum(r["dylan"] for r in rows)
    wins = sum(1 for r in rows if r["margin"] > 0); losses = sum(1 for r in rows if r["margin"] < 0)
    def rate(key, side, idx=0):
        num = sum(r[key][side][idx] for r in rows); den = sum(r[key][side][1] for r in rows)
        return (num / den if den else 0.0), den
    out = {"variant": VARIANT, "games": n, "margin": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se],
           "kv_sets": kv_sets, "dylan_sets": dy_sets, "kv_set_share": kv_sets / max(1, kv_sets + dy_sets),
           "kv_win_pct": 100 * wins / n, "ties": n - wins - losses,
           "ask_hit_kv": rate("asks", "kv"), "ask_hit_dy": rate("asks", "dy"),
           "decl_ok_kv": rate("decl", "kv"), "decl_ok_dy": rate("decl", "dy"),
           "forced_kv": [sum(r["forced"]["kv"][0] for r in rows), sum(r["forced"]["kv"][1] for r in rows)],
           "forced_dy": [sum(r["forced"]["dy"][0] for r in rows), sum(r["forced"]["dy"][1] for r in rows)],
           "own_err_kv": sum(r["own_err"]["kv"] for r in rows), "own_err_dy": sum(r["own_err"]["dy"] for r in rows),
           "alloc_err_kv": sum(r["alloc_err"]["kv"] for r in rows), "alloc_err_dy": sum(r["alloc_err"]["dy"] for r in rows),
           "fallbacks": sum(r["fallbacks"] for r in rows), "off_team_owners": sum(r["off_team_owners"] for r in rows),
           "unfinished": sum(1 for r in rows if not r["terminal"]),
           "sec_per_game": sum(r["seconds"] for r in rows) / n}
    ex = {}
    for r in rows:
        for k, v in r.get("extra", {}).items(): ex[k] = ex.get(k, 0) + v
    out["extra"] = ex
    return out

if __name__ == "__main__":
    n_deals = int(sys.argv[1]); workers = int(sys.argv[2]); seed0 = int(sys.argv[3]); journal = sys.argv[4]
    done = set(); rows = []
    if os.path.exists(journal):
        for line in open(journal):
            if line.strip():
                r = json.loads(line)
                if r.get("variant") == VARIANT:
                    done.add((r["deal"], r["kv_even"])); rows.append(r)
    tasks = [(seed0 + i, par) for i in range(n_deals) for par in (True, False) if (seed0 + i, par) not in done]
    print(f"variant={VARIANT} {len(rows)} done, {len(tasks)} to play, workers={workers}", flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(play, tasks, chunksize=1), 1):
            rows.append(r)
            with open(journal, "a") as fh: fh.write(json.dumps(r) + "\n")
            if k % 100 == 0:
                s = summarize(rows)
                print(f"  {k}/{len(tasks)} games  margin {s['margin']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  "
                      f"declOK kv {100*s['decl_ok_kv'][0]:.1f}% dy {100*s['decl_ok_dy'][0]:.1f}%  fallbacks {s['fallbacks']}  {time.time()-t0:.0f}s", flush=True)
    s = summarize(rows)
    print(json.dumps(s, indent=1))
    json.dump(s, open(journal.replace(".jsonl", "_summary.json"), "w"), indent=1)
