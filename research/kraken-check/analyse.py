#!/usr/bin/env python3
"""Aggregate E1 (FishLab-arbiter blocks) and E2/E3 (KV-arbiter journals) into summary.json."""
import json, os, math, collections
S = os.path.dirname(os.path.abspath(__file__))

def e1():
    rows = [json.loads(l) for l in open(os.path.join(S, "e1_results.jsonl")) if l.strip()]
    cells = collections.OrderedDict()
    for r in rows: cells.setdefault(r["cell"], []).append(r)
    print("E1 -- FishLab arbiter, blocks of 100 deals x 6 rotations (600 games); CI = Student t on block means (deal-clustered)")
    print(f"{'cell':18} {'blocks':>6} {'games':>6} {'A win%':>7} {'95% CI':>16} {'sets A-B':>9} {'95% CI':>16} {'declA%':>7} {'declB%':>7} {'askA%':>6} {'askB%':>6} {'forcedA/g':>9}")
    out = collections.OrderedDict()
    for c, rs in cells.items():
        n = len(rs); g = sum(r["games"] for r in rs)
        w = [r["winRateA"] for r in rs]; m = [r["meanSetsA"] - r["meanSetsB"] for r in rs]
        def ci(x):
            # Student t on the block means (few blocks), not a normal quantile.
            T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
            mu = sum(x) / len(x)
            if len(x) < 2: return mu, float("nan"), float("nan")
            sd = math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1)); h = T.get(len(x) - 1, 1.96) * sd / math.sqrt(len(x))
            return mu, mu - h, mu + h
        wm, wl, wh = ci(w); mm, ml, mh = ci(m)
        dA = sum(r["declAccA"] * r["nDeclA"] for r in rs) / max(1, sum(r["nDeclA"] for r in rs))
        dB = sum(r["declAccB"] * r["nDeclB"] for r in rs) / max(1, sum(r["nDeclB"] for r in rs))
        aA = sum(r["askAccA"] * r["nAsksA"] for r in rs) / max(1, sum(r["nAsksA"] for r in rs))
        aB = sum(r["askAccB"] * r["nAsksB"] for r in rs) / max(1, sum(r["nAsksB"] for r in rs))
        fA = sum(r["forcedPerGameA"] for r in rs) / n
        print(f"{c:18} {n:6d} {g:6d} {100*wm:7.2f} [{100*wl:6.2f},{100*wh:6.2f}] {mm:+9.3f} [{ml:+7.3f},{mh:+7.3f}] {100*dA:7.2f} {100*dB:7.2f} {100*aA:6.2f} {100*aB:6.2f} {fA:9.3f}")
        out[c] = {"blocks": n, "games": g, "winA": wm, "winA_ci": [wl, wh], "sets_margin_A": mm, "sets_margin_ci": [ml, mh],
                  "declA": dA, "declB": dB, "askA": aA, "askB": aB, "a": rs[0]["a"], "b": rs[0]["b"], "flags": rs[0]["flags"]}
    return out

def load(v):
    p = os.path.join(S, f"e2_{v}.jsonl")
    if not os.path.exists(p): return {}
    return {(r["deal"], r["kv_even"]): r for r in (json.loads(l) for l in open(p) if l.strip())}

VARIANTS = ["rev2", "dealthand", "ladder", "seedvary", "dealtseed", "dealthand_big"]

def e2():
    base = load("rev2")
    print("\nE2/E3 -- KV's arbiter (their engine, their rules, their bridge), 1000 deals x 2 parities, identical deals across variants")
    print(f"{'variant':13} {'games':>5} {'KV margin':>10} {'95% CI':>16} {'KV set%':>7} {'KV win%':>7} {'S decl%':>7} {'S vol%':>7} {'S forced%':>9} {'S own_err/g':>11} {'S alloc_err/g':>13} {'paired d vs rev2':>24} {'fallbacks':>9}")
    out = collections.OrderedDict()
    for v in VARIANTS:
        rows = load(v)
        if not rows: continue
        rs = list(rows.values()); n = len(rs)
        m = [r["margin"] for r in rs]; mu = sum(m) / n; sd = math.sqrt(sum((x - mu) ** 2 for x in m) / max(1, n - 1)); h = 1.96 * sd / math.sqrt(n)
        kvs = sum(r["kv"] for r in rs); dys = sum(r["dylan"] for r in rs)
        wins = sum(1 for r in rs if r["margin"] > 0)
        dd = [sum(r["decl"]["dy"][0] for r in rs), sum(r["decl"]["dy"][1] for r in rs)]
        fd = [sum(r["forced"]["dy"][1] for r in rs), sum(r["forced"]["dy"][0] for r in rs)]
        vol = [(dd[0] - fd[0]), (dd[1] - fd[1])]
        dk = [sum(r["decl"]["kv"][0] for r in rs), sum(r["decl"]["kv"][1] for r in rs)]
        ak = [sum(r["asks"]["kv"][0] for r in rs), sum(r["asks"]["kv"][1] for r in rs)]
        ad = [sum(r["asks"]["dy"][0] for r in rs), sum(r["asks"]["dy"][1] for r in rs)]
        own = sum(r["own_err"]["dy"] for r in rs) / n; alloc = sum(r["alloc_err"]["dy"] for r in rs) / n
        paired = ""; pd = None
        if v != "rev2" and base:
            keys = [k for k in rows if k in base]
            d = [rows[k]["margin"] - base[k]["margin"] for k in keys]
            if len(d) > 1:
                dm = sum(d) / len(d); dsd = math.sqrt(sum((x - dm) ** 2 for x in d) / (len(d) - 1)); dh = 1.96 * dsd / math.sqrt(len(d))
                paired = f"{dm:+.3f} [{dm-dh:+.3f},{dm+dh:+.3f}]"; pd = [dm, dm - dh, dm + dh, len(d)]
        fb = sum(r["fallbacks"] for r in rs)
        print(f"{v:13} {n:5d} {mu:+10.3f} [{mu-h:+7.3f},{mu+h:+7.3f}] {100*kvs/(kvs+dys):7.2f} {100*wins/n:7.2f} {100*dd[0]/dd[1]:7.2f} {100*vol[0]/max(1,vol[1]):7.2f} {100*fd[0]/max(1,fd[1]):9.2f} {own:11.3f} {alloc:13.3f} {paired:>24} {fb:9d}")
        ex = {}
        for r in rs:
            for k2, v2 in r.get("extra", {}).items(): ex[k2] = ex.get(k2, 0) + v2
        if ex: print(f"{'':13} extra: {ex}")
        out[v] = {"games": n, "margin": mu, "ci": [mu - h, mu + h], "kv_set_share": kvs / (kvs + dys), "kv_win": wins / n,
                  "sestina_decl_acc": dd[0] / dd[1], "sestina_voluntary_acc": vol[0] / max(1, vol[1]), "sestina_forced_acc": fd[0] / max(1, fd[1]),
                  "sestina_forced_per_game": fd[1] / n, "kraken_decl_acc": dk[0] / max(1, dk[1]), "kraken_ask_hit": ak[0] / max(1, ak[1]), "sestina_ask_hit": ad[0] / max(1, ad[1]),
                  "own_err_per_game": own, "alloc_err_per_game": alloc, "paired_vs_rev2": pd, "fallbacks": fb,
                  "unfinished": sum(1 for r in rs if not r["terminal"]), "extra": ex}
    return out

if __name__ == "__main__":
    res = {"e1": e1(), "e2": e2()}
    json.dump(res, open(os.path.join(S, "summary.json"), "w"), indent=1)
