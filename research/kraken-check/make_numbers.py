#!/usr/bin/env python3
"""Generate paper/kraken_check_numbers.tex (one macro per number) and paper/tables_kraken/*.tex
from the artifacts in research/kraken-check/.  The paper contains no typed digits from the
measurements: every figure is a macro defined here, with its source artifact named."""
import json, os, sys, math, io, contextlib
A = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(A, "..", ".."))
sys.path.insert(0, A)
import analyse
with contextlib.redirect_stdout(io.StringIO()):
    E1 = analyse.e1(); E2 = analyse.e2()
mega = json.load(open(os.path.join(A, "upstream", "results", "mega_match.json")))
probe = json.load(open(os.path.join(A, "e5_probe.json")))
up = json.loads([l for l in open(os.path.join(A, "e4_upstream_777001.json")) if l.startswith("{")][-1])
e1rows = [json.loads(l) for l in open(os.path.join(A, "e1_results.jsonl")) if l.strip()]
vend = [r for r in e1rows if r["cell"] == "default" and r["seed"] == 777001][0]
br = json.load(open(os.path.join(A, "bridge_kraken_stats.json")))
uj = [json.loads(l) for l in open(os.path.join(A, "upstream", "results", "mega_match_journal.jsonl")) if l.strip()]
upstream_head = open(os.path.join(A, "upstream", "UPSTREAM.txt")).read().split()[-2]

M = []   # (name, value, source)
def m(name, value, src): M.append((name, value, src))
def f1(x): return f"{100*x:.1f}"
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:+.3f}"
def f4(x): return f"{x:+.4f}"
def n(x): return f"{x:,}"

# --- the claim (transcribed from KV's artifacts) ---
m("MegaMargin", f4(mega["margin"]), "upstream/results/mega_match.json")
m("MegaLo", f4(mega["ci95"][0]), "mega_match.json"); m("MegaHi", f4(mega["ci95"][1]), "mega_match.json")
m("MegaSD", f2(mega["sd"]), "mega_match.json")
m("MegaSetShare", f1(mega["kv_set_share"]), "mega_match.json"); m("MegaWin", f1(mega["win_rate"]), "mega_match.json")
m("MegaWins", n(mega["wins"]), "mega_match.json"); m("MegaLosses", n(mega["losses"]), "mega_match.json")
m("MegaN", n(mega["n_games"]), "mega_match.json")
m("MegaDeclKV", f1(mega["declare_right_kv"]), "mega_match.json"); m("MegaDeclS", f1(mega["declare_right_dylan"]), "mega_match.json")
m("MegaAskKV", f1(mega["ask_hit_kv"]), "mega_match.json"); m("MegaAskS", f1(mega["ask_hit_dylan"]), "mega_match.json")
m("MegaWrongS", n(mega["wrong_declarations_dylan"]), "mega_match.json"); m("MegaWrongKV", n(mega["wrong_declarations_kv"]), "mega_match.json")
m("MegaAllocS", n(mega["allocation_misdeclares_dylan"]), "mega_match.json")
m("MegaBridgeRev", str(mega["bridge_rev"]), "mega_match.json")
nj = len(uj)
m("UjGames", n(nj), "upstream/results/mega_match_journal.jsonl")
m("UjDeclPerGameKV", f2(sum(r["dec"]["kv"][1] for r in uj) / nj), "mega_match_journal.jsonl")
m("UjDeclPerGameS", f2(sum(r["dec"]["dy"][1] for r in uj) / nj), "mega_match_journal.jsonl")
m("UjWrongPerGameS", f3(sum(r["dec"]["dy"][1] - r["dec"]["dy"][0] for r in uj) / nj).lstrip("+"), "mega_match_journal.jsonl")
m("UjWrongPerGameKV", f3(sum(r["dec"]["kv"][1] - r["dec"]["kv"][0] for r in uj) / nj).lstrip("+"), "mega_match_journal.jsonl")
m("UjUnfinished", str(sum(1 for r in uj if not r["terminal"])), "mega_match_journal.jsonl")
m("UjFallbacks", str(sum(r["fallbacks"] for r in uj)), "mega_match_journal.jsonl")
m("UpstreamHead", upstream_head[:7], "upstream/UPSTREAM.txt")
# KV's package README dialect table (transcribed; kraken/README.md, "Three things a host MUST get right")
# Figures transcribed from the KRAKEN v1.1 paper (kraken.pdf), section 11.9 and the abstract
m("PaperNeverMoved", "398", "kraken.pdf s11.9 (transcribed)"); m("PaperRateRatio", "67", "kraken.pdf s11.9 (transcribed)")
m("PaperFollowedBoth", "1,927", "kraken.pdf s11.9 (transcribed)"); m("PaperFollowedBothPct", "19.25", "kraken.pdf s11.9 (transcribed)"); m("PaperFollowedTakingPct", "3.16", "kraken.pdf s11.9 (transcribed)")
m("PaperDeclShare", "57", "kraken.pdf s11.9 (transcribed)"); m("PaperTheirWrongPerGame", "0.844", "kraken.pdf s11.9 (transcribed)")
m("PaperOurWrongPerGame", "0.176", "kraken.pdf s11.9 (transcribed)"); m("PaperBridgeBugPrice", "-0.0784", "kraken.pdf s11.9 (transcribed)")
m("PaperVthreeMargin", "+2.412", "kraken.pdf abstract (transcribed)"); m("PaperShimDecisions", "329", "kraken.pdf appendix C (transcribed)")
m("BeliefCertLines", "189--201", "engine/src/belief.hpp"); m("BeliefPinLine", "200", "engine/src/belief.hpp")
m("ReadmeOffturnGap", "-0.300", "derived from the README table: both-may minus nobody")
m("EOffturnGap", f3(E1["default"]["sets_margin_A"] - E1["nooot"]["sets_margin_A"]), "derived: default minus on-turn-only")
m("ReadmeNobody", "+2.675", "engine/external/kraken/kraken/README.md (transcribed)")
m("ReadmeBoth", "+2.375", "kraken/README.md (transcribed)")
m("ReadmeOnlyYou", "+1.575", "kraken/README.md (transcribed)")

# --- E1 cells ---
CELL = {"default": "Default", "nooot": "Nooot", "mirror_kraken": "Mirror", "kv6_vs_v07": "KvSix", "kraken_vs_v06": "VsVsix",
        "nooot_nocardless": "NoootNc", "legacy": "Legacy", "kraken_vs_fcheap": "VsFcheap", "kraken_vs_v05": "VsVfive", "kraken_vs_v04": "VsVfour", "kv6_vs_v07_paired": "KvSixPaired"}
for c, key in CELL.items():
    if c not in E1: continue
    r = E1[c]
    m(f"E{key}Games", n(r["games"]), "e1_results.jsonl"); m(f"E{key}Blocks", str(r["blocks"]), "e1_results.jsonl")
    m(f"E{key}Win", f1(r["winA"]), "e1_results.jsonl")
    m(f"E{key}WinLo", f1(r["winA_ci"][0]), "e1_results.jsonl"); m(f"E{key}WinHi", f1(r["winA_ci"][1]), "e1_results.jsonl")
    m(f"E{key}Margin", f3(r["sets_margin_A"]), "e1_results.jsonl")
    m(f"E{key}MarginLo", f3(r["sets_margin_ci"][0]), "e1_results.jsonl"); m(f"E{key}MarginHi", f3(r["sets_margin_ci"][1]), "e1_results.jsonl")
    m(f"E{key}DeclA", f1(r["declA"]), "e1_results.jsonl"); m(f"E{key}DeclB", f1(r["declB"]), "e1_results.jsonl")
    m(f"E{key}AskA", f1(r["askA"]), "e1_results.jsonl"); m(f"E{key}AskB", f1(r["askB"]), "e1_results.jsonl")
sest = ["default", "nooot", "nooot_nocardless", "legacy"]
m("ESestinaGames", n(sum(E1[c]["games"] for c in sest)), "e1_results.jsonl")
m("ESestinaBlocks", str(sum(E1[c]["blocks"] for c in sest)), "e1_results.jsonl")
m("ESestinaWinMin", f1(min(E1[c]["winA"] for c in sest)), "e1_results.jsonl"); m("ESestinaWinMax", f1(max(E1[c]["winA"] for c in sest)), "e1_results.jsonl")
m("ESestinaMarginMin", f3(min(E1[c]["sets_margin_A"] for c in sest)), "e1_results.jsonl"); m("ESestinaMarginMax", f3(max(E1[c]["sets_margin_A"] for c in sest)), "e1_results.jsonl")
m("ETotalGames", n(sum(r["games"] for r in E1.values())), "e1_results.jsonl"); m("ETotalBlocks", str(sum(r["blocks"] for r in E1.values())), "e1_results.jsonl")
# standard errors from the claim: how many SEs the default cell sits from the paper's figure
TQ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
for c_, key in (("default", "Default"), ("nooot", "Nooot")):
    se = (E1[c_]["sets_margin_ci"][1] - E1[c_]["sets_margin_A"]) / TQ[E1[c_]["blocks"] - 1]
    m(f"E{key}SEsFromClaim", str(int((mega["margin"] - E1[c_]["sets_margin_A"]) / se)), "derived: (claim - cell) / block SE")
# cardless declaration, KRAKEN's dialect with and without it (unpaired, different deals)
m("ECardlessShift", f3(E1["nooot_nocardless"]["sets_margin_A"] - E1["nooot"]["sets_margin_A"]), "derived: nooot_nocardless - nooot")
# KV FishBot v0.6 on the KRAKEN default cell's seeds: block-paired difference
if "kv6_vs_v07_paired" in E1:
    kr = {r["seed"]: r for r in e1rows if r["cell"] == "default"}
    kv = {r["seed"]: r for r in e1rows if r["cell"] == "kv6_vs_v07_paired"}
    d = [ (kv[sd]["meanSetsA"] - kv[sd]["meanSetsB"]) - (kr[sd]["meanSetsA"] - kr[sd]["meanSetsB"]) for sd in kv if sd in kr ]
    mu = sum(d) / len(d); sdv = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1)); h = TQ[len(d) - 1] * sdv / math.sqrt(len(d))
    m("EKvSixDiffBlocks", str(len(d)), "e1_results.jsonl"); m("EKvSixDiff", f3(mu), "derived: kv6 - kraken on identical blocks")
    m("EKvSixDiffLo", f3(mu - h), "derived"); m("EKvSixDiffHi", f3(mu + h), "derived")
# forced-fallback bound for KRAKEN in FishLab: forced declarations per game x forced error rate x 2 sets
for c_, key in (("default", "Default"), ("nooot", "Nooot")):
    rs = [r for r in e1rows if r["cell"] == c_]
    fpg = sum(r["forcedPerGameA"] for r in rs) / len(rs)
    facc = sum(r["forcedAccA"] * r["forcedPerGameA"] * r["games"] for r in rs) / max(1e-9, sum(r["forcedPerGameA"] * r["games"] for r in rs))
    m(f"E{key}ForcedPerGame", f"{fpg:.3f}", "e1_results.jsonl"); m(f"E{key}ForcedAcc", f1(facc), "e1_results.jsonl")
    m(f"E{key}FallbackBound", f"{2 * fpg * (1 - facc):.3f}", "derived: 2 x forced/game x (1 - forced accuracy)")
# deal-clustered check of the KV-arbiter intervals: the two parities of a deal as a cluster
def clustered(rows_by_key):
    import collections
    by = collections.defaultdict(list)
    for (deal, par), r in rows_by_key.items(): by[deal].append(r["margin"])
    deals = [v for v in by.values() if len(v) == 2]
    a = [v[0] for v in deals]; b = [v[1] for v in deals]
    ma = sum(a) / len(a); mb = sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1); vb = sum((y - mb) ** 2 for y in b) / (len(b) - 1)
    rho = cov / math.sqrt(va * vb)
    sums = [x + y for x, y in zip(a, b)]; ms = sum(sums) / len(sums)
    vs = sum((x - ms) ** 2 for x in sums) / (len(sums) - 1)
    half_cl = 1.96 * math.sqrt(vs / len(sums)) / 2          # half-width of the per-game mean, clustering by deal
    allm = a + b; mm = sum(allm) / len(allm); vv = sum((x - mm) ** 2 for x in allm) / (len(allm) - 1)
    half_ind = 1.96 * math.sqrt(vv / len(allm))
    return rho, half_cl, half_ind
maxrho = 0; maxdiff = 0
for v in ["rev2", "dealthand", "ladder", "seedvary", "dealtseed", "dealthand_big"]:
    rows_v = analyse.load(v)
    if not rows_v: continue
    rho, hc, hi = clustered(rows_v); maxrho = max(maxrho, abs(rho)); maxdiff = max(maxdiff, abs(hc - hi))
m("VMaxAbsRho", f"{maxrho:.3f}", "derived: parity correlation within a deal, worst over variants"); m("VMaxClusterDiff", f"{maxdiff:.3f}", "derived: |clustered - unclustered half-width|, worst over variants")
gps = sorted(r["gamesPerSec"] for r in e1rows if r["cell"] == "default")
m("EThroughput", f"{gps[len(gps)//2]:.0f}", "e1_results.jsonl (median block, 9 threads shared with other runs)")
m("EThreads", str(e1rows[0]["threads"]), "e1_results.jsonl")

# --- E2 variants ---
VAR = {"rev2": "RevTwo", "dealthand": "Dealt", "ladder": "Ladder", "seedvary": "Seed", "dealtseed": "DealtSeed", "dealthand_big": "DealtBig"}
for v, key in VAR.items():
    r = E2[v]
    m(f"V{key}Games", n(r["games"]), f"e2_{v}.jsonl")
    m(f"V{key}Margin", f3(r["margin"]), f"e2_{v}.jsonl"); m(f"V{key}Lo", f3(r["ci"][0]), f"e2_{v}.jsonl"); m(f"V{key}Hi", f3(r["ci"][1]), f"e2_{v}.jsonl")
    m(f"V{key}SetShare", f1(r["kv_set_share"]), f"e2_{v}.jsonl"); m(f"V{key}Win", f1(r["kv_win"]), f"e2_{v}.jsonl")
    m(f"V{key}DeclS", f1(r["sestina_decl_acc"]), f"e2_{v}.jsonl"); m(f"V{key}VolS", f1(r["sestina_voluntary_acc"]), f"e2_{v}.jsonl")
    m(f"V{key}ForcedS", f1(r["sestina_forced_acc"]), f"e2_{v}.jsonl"); m(f"V{key}ForcedPerGame", f2(r["sestina_forced_per_game"]), f"e2_{v}.jsonl")
    m(f"V{key}DeclKV", f1(r["kraken_decl_acc"]), f"e2_{v}.jsonl"); m(f"V{key}AskS", f1(r["sestina_ask_hit"]), f"e2_{v}.jsonl"); m(f"V{key}AskKV", f1(r["kraken_ask_hit"]), f"e2_{v}.jsonl")
    m(f"V{key}OwnErr", f"{r['own_err_per_game']:.3f}", f"e2_{v}.jsonl"); m(f"V{key}AllocErr", f"{r['alloc_err_per_game']:.3f}", f"e2_{v}.jsonl")
    m(f"V{key}Fallbacks", str(r["fallbacks"]), f"e2_{v}.jsonl"); m(f"V{key}Unfinished", str(r["unfinished"]), f"e2_{v}.jsonl")
    if r["paired_vs_rev2"]:
        pd = r["paired_vs_rev2"]
        m(f"V{key}Paired", f3(pd[0]), f"e2_{v}.jsonl vs e2_rev2.jsonl"); m(f"V{key}PairedLo", f3(pd[1]), "paired"); m(f"V{key}PairedHi", f3(pd[2]), "paired"); m(f"V{key}PairedN", n(pd[3]), "paired")
m("VDealtHandMismatch", str(E2["dealthand"]["extra"].get("hand_mismatch", 0)), "e2_dealthand.jsonl")
m("VDealtBigHandMismatch", str(E2["dealthand_big"]["extra"].get("hand_mismatch", 0)), "e2_dealthand_big.jsonl")
m("VLadderCalls", n(E2["ladder"]["extra"].get("ladder_calls", 0)), "e2_ladder.jsonl")
m("VRevTwoWrongPct", f1(1 - E2["rev2"]["sestina_decl_acc"]), "e2_rev2.jsonl")
m("VRevTwoVolWrongPct", f1(1 - E2["rev2"]["sestina_voluntary_acc"]), "e2_rev2.jsonl")

# --- probe ---
c, d = probe["cur"], probe["dealt"]
m("PrGames", str(probe["games"]), "e5_probe.json"); m("PrDecisions", n(probe["onTurnDecisions"]), "e5_probe.json")
m("PrNativeSets", str(probe["setsNative"]), "e5_probe.json"); m("PrOppSets", str(probe["setsOpp"]), "e5_probe.json"); m("PrNativeWins", str(probe["winsNative"]), "e5_probe.json")
m("PrNativeDecl", str(probe["innerDecl"]), "e5_probe.json"); m("PrNativeWrong", str(probe["innerDeclWrong"]), "e5_probe.json")
m("PrNativeWrongPct", f1(probe["innerDeclWrong"] / probe["innerDecl"]), "e5_probe.json")
for key, x in (("Cur", c), ("Dealt", d)):
    m(f"Pr{key}Decl", str(x["decl"]), "e5_probe.json"); m(f"Pr{key}Wrong", str(x["declWrong"]), "e5_probe.json")
    m(f"Pr{key}WrongPct", f1(x["declWrong"] / x["decl"]), "e5_probe.json")
    m(f"Pr{key}Agree", n(x["agreeDecl"]), "e5_probe.json"); m(f"Pr{key}AgreePct", f1(x["agreeDecl"] / probe["onTurnDecisions"]), "e5_probe.json")
    m(f"Pr{key}Only", str(x["shimOnly"]), "e5_probe.json"); m(f"Pr{key}OnlyWrong", str(x["shimOnlyWrong"]), "e5_probe.json"); m(f"Pr{key}OnlyOwn", str(x["shimOnlyOwnershipWrong"]), "e5_probe.json")
    m(f"Pr{key}NativeOnly", str(x["nativeOnly"]), "e5_probe.json"); m(f"Pr{key}BothDiff", str(x["bothDiffer"]), "e5_probe.json")
    m(f"Pr{key}Ask", n(x["askAgree"]), "e5_probe.json"); m(f"Pr{key}AskPct", f1(x["askAgree"] / probe["askCompared"]), "e5_probe.json")
m("PrAskCompared", n(probe["askCompared"]), "e5_probe.json")
fp = os.path.join(A, "e5_probe_fresh.json")
if os.path.exists(fp) and os.path.getsize(fp) > 0:
    pf = json.load(open(fp))
    m("PrFreshFirst", str(pf["firstDecisions"]), "e5_probe_fresh.json"); m("PrFreshFirstCur", str(pf["firstAgreeCur"]), "e5_probe_fresh.json"); m("PrFreshFirstDealt", str(pf["firstAgreeDealt"]), "e5_probe_fresh.json")
    m("PrFreshDealtWrongPct", f1(pf["dealt"]["declWrong"] / pf["dealt"]["decl"]), "e5_probe_fresh.json"); m("PrFreshNativeWrongPct", f1(pf["innerDeclWrong"] / pf["innerDecl"]), "e5_probe_fresh.json")
    m("PrFreshDealtOnly", str(pf["dealt"]["shimOnly"]), "e5_probe_fresh.json"); m("PrFreshDealtAskPct", f1(pf["dealt"]["askAgree"] / pf["askCompared"]), "e5_probe_fresh.json")
    m("PrFreshDealtNativeOnly", str(pf["dealt"]["nativeOnly"]), "e5_probe_fresh.json"); m("PrFreshDealtBothDiff", str(pf["dealt"]["bothDiffer"]), "e5_probe_fresh.json"); m("PrFirst", str(probe["firstDecisions"]), "e5_probe.json")
m("PrFirstCur", str(probe["firstAgreeCur"]), "e5_probe.json"); m("PrFirstDealt", str(probe["firstAgreeDealt"]), "e5_probe.json"); m("PrShimErrors", str(probe["shimErrors"]), "e5_probe.json")

# --- upstream identity ---
ident = all(abs(up[k] - vend[k]) < 1e-9 for k in ("winRateA", "meanSetsA", "meanSetsB")) and all(up[k] == vend[k] for k in ("nAsksA", "nAsksB", "nDeclA", "nDeclB"))
m("IdIdentical", "identical" if ident else "NOT identical", "e4_upstream_777001.json vs e1_results.jsonl seed 777001")
m("IdWin", f1(vend["winRateA"]), "e1_results.jsonl seed 777001"); m("IdUpWin", f1(up["winRateA"]), "e4_upstream_777001.json")
m("IdSetsA", f"{vend['meanSetsA']:.3f}", "e1"); m("IdSetsB", f"{vend['meanSetsB']:.3f}", "e1"); m("IdUpSetsA", f"{up['meanSetsA']:.3f}", "e4"); m("IdUpSetsB", f"{up['meanSetsB']:.3f}", "e4")
m("IdAsks", n(vend["nAsksA"]), "e1"); m("IdUpAsks", n(up["nAsksA"]), "e4"); m("IdDecl", n(vend["nDeclA"]), "e1"); m("IdUpDecl", n(up["nDeclA"]), "e4")
# --- bridge transcript ---
m("BrRequests", str(br["requests"]), "bridge_kraken_stats.json"); m("BrDecide", str(br["ops"]["decide"]), "bridge_kraken_stats.json")
m("BrOffturn", str(br["ops"]["offturn"]), "bridge_kraken_stats.json"); m("BrErrors", str(br["error_replies"]), "bridge_kraken_stats.json")

out = ["% Generated by research/kraken-check/make_numbers.py -- do not edit.  One macro per number;",
       "% the comment on each line names the artifact it was read from."]
for name, value, src in M:
    out.append(f"\\newcommand{{\\kc{name}}}{{{value}}}  % {src}")
open(os.path.join(R, "paper", "kraken_check_numbers.tex"), "w").write("\n".join(out) + "\n")
bl = open(os.path.join(R, "engine", "src", "belief.hpp")).read().split("\n")
excerpt = bl[188:201]    # lines 189-201 inclusive, verbatim
os.makedirs(os.path.join(R, "paper", "tables_kraken"), exist_ok=True)
open(os.path.join(R, "paper", "tables_kraken", "belief_excerpt.tex"), "w").write("\\begin{lstlisting}\n" + "\n".join(l[4:] if l.startswith("    ") else l for l in excerpt) + "\n\\end{lstlisting}\n")

# ------------------------------------------------------------------ tables
T = os.path.join(R, "paper", "tables_kraken"); os.makedirs(T, exist_ok=True)
def K(name): return "\\kc" + name + "{}"

# Table: the claim against the measurements
rows = [
 ("KRAKEN margin, sets per game", f"${K('MegaMargin')}$ $[{K('MegaLo')}, {K('MegaHi')}]$", f"${K('EDefaultMargin')}$ $[{K('EDefaultMarginLo')}, {K('EDefaultMarginHi')}]$", f"${K('VDealtBigMargin')}$ $[{K('VDealtBigLo')}, {K('VDealtBigHi')}]$"),
 ("KRAKEN share of decided sets", f"{K('MegaSetShare')}\\%", f"{100*E1['default']['a'] and 0:.0f}", f"{K('VDealtBigSetShare')}\\%"),
 ("KRAKEN games won", f"{K('MegaWin')}\\%", f"{K('EDefaultWin')}\\%", f"{K('VDealtBigWin')}\\%"),
 ("SESTINA declaration accuracy", f"{K('MegaDeclS')}\\%", f"{K('EDefaultDeclB')}\\%", f"{K('VDealtBigDeclS')}\\%"),
 ("KRAKEN declaration accuracy", f"{K('MegaDeclKV')}\\%", f"{K('EDefaultDeclA')}\\%", f"{K('VDealtBigDeclKV')}\\%"),
 ("SESTINA ask hit rate", f"{K('MegaAskS')}\\%", f"{K('EDefaultAskB')}\\%", f"{K('VDealtBigAskS')}\\%"),
 ("KRAKEN ask hit rate", f"{K('MegaAskKV')}\\%", f"{K('EDefaultAskA')}\\%", f"{K('VDealtBigAskKV')}\\%"),
 ("games", K("MegaN"), K("EDefaultGames"), K("VDealtBigGames")),
]
# set share for FishLab default cell: sets A / (A+B) from e1 rows
tot_a = sum(r["meanSetsA"] * r["games"] for r in e1rows if r["cell"] == "default"); tot_b = sum(r["meanSetsB"] * r["games"] for r in e1rows if r["cell"] == "default")
M.append(("EDefaultSetShare", f1(tot_a / (tot_a + tot_b)), "e1_results.jsonl"))
open(os.path.join(R, "paper", "kraken_check_numbers.tex"), "a").write(f"\\newcommand{{\\kcEDefaultSetShare}}{{{f1(tot_a / (tot_a + tot_b))}}}  % e1_results.jsonl\n")
rows[1] = ("KRAKEN share of decided sets", f"{K('MegaSetShare')}\\%", f"{K('EDefaultSetShare')}\\%", f"{K('VDealtBigSetShare')}\\%")
L = ["\\begin{tabularx}{\\linewidth}{Q{1.0}Q{1.0}Q{1.0}Q{1.0}}", "\\toprule",
     "quantity & the paper (KV's arbiter, their bridge) & FishLab's arbiter, FishLab rules & KV's arbiter, bridge repaired \\\\", "\\midrule"]
for q, a, b, c_ in rows: L.append(f"{q} & {a} & {b} & {c_} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
open(os.path.join(T, "claims.tex"), "w").write("\n".join(L) + "\n")

# Table: FishLab arbiter cells
names = {"default": ("KRAKEN", "SESTINA", "FishLab default"), "nooot": ("KRAKEN", "SESTINA", "declare on turn only"),
         "nooot_nocardless": ("KRAKEN", "SESTINA", "on turn only, no cardless declaration"), "legacy": ("KRAKEN", "SESTINA", "legacy dialect"),
         "mirror_kraken": ("KRAKEN", "KRAKEN", "FishLab default (control)"), "kv6_vs_v07": ("KV FishBot v0.6", "SESTINA", "FishLab default"),
         "kraken_vs_v06": ("KRAKEN", "FishBot v0.6", "FishLab default"), "kraken_vs_fcheap": ("KRAKEN", "F-cheap", "FishLab default"),
         "kraken_vs_v05": ("KRAKEN", "FishBot v0.5", "FishLab default"), "kraken_vs_v04": ("KRAKEN", "FishBot v0.4", "FishLab default"), "kv6_vs_v07_paired": ("KV FishBot v0.6", "SESTINA", "FishLab default, the KRAKEN cell's seeds")}
L = ["\\begin{tabularx}{\\linewidth}{Q{0.95}Q{1.0}Q{1.45}N{0.75}N{0.8}N{1.15}N{0.9}}", "\\toprule",
     "A & B & rules & games & A wins & sets A$-$B per game & decl.\\ acc.\\ A\\,/\\,B (\\%) \\\\", "\\midrule"]
ORDER = ["default", "nooot", "nooot_nocardless", "legacy", "|", "mirror_kraken", "|", "kraken_vs_v06", "kraken_vs_fcheap", "kraken_vs_v05", "kraken_vs_v04", "|", "kv6_vs_v07", "kv6_vs_v07_paired"]
for c_ in ORDER:
    if c_ == "|": L.append("\\midrule"); continue
    if c_ not in E1: continue
    key = CELL[c_]
    a, b, ru = names[c_]
    L.append(f"{a} & {b} & {ru} & {K('E'+key+'Games')} & {K('E'+key+'Win')}\\% & \\ci{{${K('E'+key+'Margin')}$}}{{{K('E'+key+'MarginLo')},\\,{K('E'+key+'MarginHi')}}} & {K('E'+key+'DeclA')}\\,/\\,{K('E'+key+'DeclB')} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
open(os.path.join(T, "fishlab.tex"), "w").write("\n".join(L) + "\n")

# Table: KV arbiter variants
desc = {"rev2": "as published (rev.~2)", "dealthand": "\\textbf{hand as dealt} instead of current hand",
        "ladder": "rev.~2 + forced-endgame ladder only", "seedvary": "rev.~2 + per-decision seed only",
        "dealtseed": "dealt hand and per-decision seed", "dealthand_big": "dealt hand, fresh deals (unpaired)"}
L = ["\\begin{tabularx}{\\linewidth}{Q{1.6}N{0.75}N{1.25}N{0.8}N{0.75}N{0.7}N{0.85}N{1.3}}", "\\toprule",
     "bridge & games & K margin & K wins & S decl.\\ acc. & S ask hit & S own.\\ err.\\ per game & paired $\\Delta$ vs rev.~2 \\\\", "\\midrule"]
for v, key in VAR.items():
    pd = f"\\ci{{${K('V'+key+'Paired')}$}}{{{K('V'+key+'PairedLo')},\\,{K('V'+key+'PairedHi')}}}" if E2[v]["paired_vs_rev2"] else "---"
    L.append(f"{desc[v]} & {K('V'+key+'Games')} & \\ci{{${K('V'+key+'Margin')}$}}{{{K('V'+key+'Lo')},\\,{K('V'+key+'Hi')}}} & {K('V'+key+'Win')}\\% & {K('V'+key+'DeclS')}\\% & {K('V'+key+'AskS')}\\% & {K('V'+key+'OwnErr')} & {pd} \\\\")
L += ["\\bottomrule", "\\end{tabularx}"]
open(os.path.join(T, "kvarena.tex"), "w").write("\n".join(L) + "\n")

# Table: probe
L = ["\\begin{tabularx}{\\linewidth}{Q{1.75}N{0.65}N{0.8}N{0.8}}", "\\toprule",
     " & native SESTINA & shim, current hand (rev.~2) & shim, hand as dealt \\\\", "\\midrule",
     f"declarations over these decisions & {K('PrNativeDecl')} & {K('PrCurDecl')} & {K('PrDealtDecl')} \\\\",
     f"of which wrong & {K('PrNativeWrong')} ({K('PrNativeWrongPct')}\\%) & {K('PrCurWrong')} ({K('PrCurWrongPct')}\\%) & {K('PrDealtWrong')} ({K('PrDealtWrongPct')}\\%) \\\\",
     f"agrees with native on declare\\,/\\,not & --- & {K('PrCurAgree')} ({K('PrCurAgreePct')}\\%) & {K('PrDealtAgree')} ({K('PrDealtAgreePct')}\\%) \\\\",
     f"declares when native does not & --- & {K('PrCurOnly')} & {K('PrDealtOnly')} \\\\",
     f"\\quad of which wrong & --- & {K('PrCurOnlyWrong')} & {K('PrDealtOnlyWrong')} \\\\",
     f"\\quad naming a half-suit an opponent holds & --- & {K('PrCurOnlyOwn')} & {K('PrDealtOnlyOwn')} \\\\",
     f"declines when native declares & --- & {K('PrCurNativeOnly')} & {K('PrDealtNativeOnly')} \\\\",
     f"both declare, different allocation & --- & {K('PrCurBothDiff')} & {K('PrDealtBothDiff')} \\\\",
     f"same ask as native (of {K('PrAskCompared')}) & --- & {K('PrCurAsk')} ({K('PrCurAskPct')}\\%) & {K('PrDealtAsk')} ({K('PrDealtAskPct')}\\%) \\\\",
     f"first decision of a game identical, agents rebuilt per game (of {K('PrFreshFirst')}) & --- & {K('PrFreshFirstCur')} & {K('PrFreshFirstDealt')} \\\\",
     f"\\quad agents reused across deals (of {K('PrFirst')}) & --- & {K('PrFirstCur')} & {K('PrFirstDealt')} \\\\",
     "\\bottomrule", "\\end{tabularx}"]
open(os.path.join(T, "probe.tex"), "w").write("\n".join(L) + "\n")

# Figure: margins across every cell, both arbiters (pgfplots), one bar per row, explicit ticks
short = {"default": "vs SESTINA, FishLab rules", "nooot": "vs SESTINA, declare on turn only", "nooot_nocardless": "vs SESTINA, on turn, no cardless",
         "legacy": "vs SESTINA, legacy dialect", "kv6_vs_v07": "KV FishBot v0.6 vs SESTINA", "kraken_vs_v06": "vs FishBot v0.6",
         "kraken_vs_fcheap": "vs F-cheap", "kraken_vs_v05": "vs FishBot v0.5", "kraken_vs_v04": "vs FishBot v0.4"}
vshort = {"rev2": "KV arbiter: bridge as published", "ladder": "KV: rev. 2 + ladder only", "seedvary": "KV: rev. 2 + seed only",
          "dealthand": "KV: hand as dealt", "dealtseed": "KV: dealt hand + seed", "dealthand_big": "KV: dealt hand, fresh deals"}
items = []
for c_ in ["default", "nooot", "nooot_nocardless", "legacy", "kraken_vs_v06", "kraken_vs_fcheap", "kraken_vs_v05", "kraken_vs_v04"]:
    r = E1[c_]; items.append(("FishLab", short[c_], r["sets_margin_A"], r["sets_margin_ci"]))
for v in ["rev2", "ladder", "seedvary", "dealthand", "dealtseed", "dealthand_big"]:
    r = E2[v]; items.append(("KV", vshort[v], r["margin"], r["ci"]))
n_it = len(items)
rows = list(enumerate(items[::-1]))   # row 0 at the bottom = last item
L = ["\\begin{tikzpicture}", "\\begin{axis}[xbar, bar shift=0pt, width=0.6\\linewidth, height=0.62\\linewidth, xmin=-1.3, xmax=3.2,",
     f"  ymin=-0.7, ymax={n_it-0.3}, ytick={{{','.join(str(i) for i in range(n_it))}}}, yticklabels={{"]
L[-1] += ", ".join("{" + it[1] + "}" for i, it in rows) + "},"
L += ["  xlabel={KRAKEN's margin, sets per game (95\\% CI)}, yticklabel style={font=\\scriptsize}, xticklabel style={font=\\scriptsize}, xlabel style={font=\\small},",
      "  bar width=5pt, axis lines*=left, xmajorgrids, grid style={gray!30}, clip=false,",
      "  error bars/x dir=both, error bars/x explicit, error bars/error bar style={black}, error bars/error mark options={rotate=90, mark size=1.5pt}]"]
for arb, color in (("FishLab", "fishgreen"), ("KV", "fishgray")):
    L.append(f"\\addplot[fill={color}, draw={color}] coordinates {{")
    for i, it in rows:
        if it[0] == arb: L.append(f"  ({it[2]:.4f},{i}) +- ({it[3][1]-it[2]:.4f},0)")
    L.append("};")
L.append("\\draw[thick] (axis cs:0,-0.7) -- (axis cs:0,%.1f);" % (n_it-0.3))
L.append("\\draw[dashed, red!70!black] (axis cs:%.4f,-0.7) -- (axis cs:%.4f,%.1f) node[above, font=\\scriptsize, text=red!70!black] {the paper's claim};" % (mega["margin"], mega["margin"], n_it-0.3))
L += ["\\end{axis}", "\\end{tikzpicture}"]
open(os.path.join(T, "margins_figure.tex"), "w").write("\n".join(L) + "\n")
print("macros:", len(M) + 1, "| tables: claims, fishlab, kvarena, probe, margins_figure")
