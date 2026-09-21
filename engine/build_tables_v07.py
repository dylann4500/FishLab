#!/usr/bin/env python3
"""SESTINA v1.0 (FishBot v0.7 cycle) -- generate every table in docs/v07/INSTRUMENT.md from the artifacts.

The project convention is that no number in a report is hand-typed: `build_tables_v06.py`
emits the v0.6 paper's macros from `research/v06/results/`, and the v0.6 audit found four
macro mis-bindings and one sign inversion in exactly that layer (SUBOPTIMALITY-LEDGER.md
P-1, P-6).  So this script prints markdown tables ready to paste, AND a JSON block of the
key scalars, from the artifacts alone.

Usage:  python3 engine/build_tables_v07.py [--dir research/v07/results] [--json]
"""
import json, os, sys, math, argparse

def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows

def hw(n):
    """The project's power rule: 95% half-width in points at p ~ 1/2."""
    return 98.0 / math.sqrt(n) if n > 0 else float('nan')

def pts(w):
    return 100.0 * (w - 0.5)

def fmt_ci(row):
    lo, hi = row['ci']
    return f"[{pts(lo):+.2f}, {pts(hi):+.2f}]"

# ---------------------------------------------------------------- throughput
def throughput(d):
    rows = load_jsonl(os.path.join(d, 'T1-throughput.jsonl'))
    # T1b supersedes T1's search block: the 50-deal timing under-samples the
    # heavy tail of per-deal search cost (RESEARCH-LOG.md 1.17), so where a
    # spec appears in both, T1b's 400-deal row is the one rendered.
    t1b = load_jsonl(os.path.join(d, 'T1b-throughput-400.jsonl'))
    if t1b:
        seen = {(r['spec'], r['mirror']) for r in t1b}
        rows = [r for r in rows if r['mirror'] or (r['spec'], r['mirror']) not in seen] + t1b
    if not rows:
        return "*(T1-throughput.jsonl absent)*\n", {}
    out = []
    blocks, cur, mirror = [], [], None
    for r in rows:
        if mirror is None or r['mirror'] != mirror:
            if cur: blocks.append((mirror, cur))
            cur, mirror = [], r['mirror']
        cur.append(r)
    if cur: blocks.append((mirror, cur))
    scal = {}
    for mir, blk in blocks:
        basis = ("mirror (each policy against itself, E9's basis)" if mir
                 else f"against `{blk[0]['opp']}` (the ledger's F-search basis)")
        out.append(f"\n**Basis: {basis}.** Reference row is the first; ratios are within the block "
                   f"and within one basis.\n")
        out.append("| configuration | games/s, all threads | games/s, 1 thread | × ref (all) | × ref (1 thr) | deals timed |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for r in blk:
            out.append(f"| `{r['spec']}` | {r['gamesPerSecAll']:.2f} | {r['gamesPerSecOne']:.3f} | "
                       f"{r['relAll']:.3f} | {r['relOne']:.3f} | {r['deals']} / {r['deals1']} |")
        for r in blk:
            scal[('mirror' if mir else 'vs') + ':' + r['spec']] = r['gamesPerSecAll']
    return "\n".join(out) + "\n", scal

# --------------------------------------------------------------- floor table
def floor(d):
    rows = load_jsonl(os.path.join(d, 'C1-floor.jsonl'))
    if not rows:
        return "*(C1-floor.jsonl absent)*\n", {}
    dtrue, resp, meta = {}, {}, {}
    for r in rows:
        if 'budget' in r and 'tag' in r and 'target' in r:
            meta[r['tag']] = r
        elif r.get('row') == 'dTrue':
            dtrue[r['tag']] = r
        elif r.get('row') in ('C1', 'C2', 'C3', 'C5'):
            resp.setdefault(r['tag'], {}).setdefault(r['row'], []).append(r)
    order = [t for t in meta] or list(dtrue)
    out = ["| target | dTrue (pts) | 95% CI | class | dFound bank A | dFound bank B | 95% CI (A) | "
           "responder decl. acc. | forced/game | limit hits |",
           "|---|---:|---|---|---:|---:|---|---:|---:|---:|"]
    scal = {}
    for tag in order:
        dt = dtrue.get(tag)
        dtv = pts(dt['winRateA']) if dt else float('nan')
        dtc = fmt_ci(dt) if dt else ''
        scal[f'dTrue:{tag}'] = dtv
        for cls in ('C1', 'C2', 'C3', 'C5'):
            rs = resp.get(tag, {}).get(cls, [])
            if not rs:
                continue
            ra = rs[0]
            rb = rs[1] if len(rs) > 1 else None
            out.append(
                f"| `{tag}` | {dtv:+.2f} | {dtc} | {cls} | {pts(ra['winRateA']):+.2f} | "
                f"{(('%+.2f' % pts(rb['winRateA'])) if rb else '—')} | {fmt_ci(ra)} | "
                f"{ra['declAccA']:.4f} | {ra['forcedPerGameA']:.4f} | {ra['limitHitRate']:.4f} |")
            scal[f'dFound:{tag}:{cls}:A'] = pts(ra['winRateA'])
            if rb:
                scal[f'dFound:{tag}:{cls}:B'] = pts(rb['winRateA'])
    return "\n".join(out) + "\n", scal

def _se_from_ci(r):
    """Approximate SE in points from a percentile-bootstrap interval."""
    return (pts(r['ci'][1]) - pts(r['ci'][0])) / (2 * 1.959963985)

def excess(d):
    """The edge a responder finds ABOVE what it finds against the unhandicapped target.

    The `none` rung is not only a false-positive control: if the incumbent is
    genuinely exploitable in a class, that class reaches a positive edge there,
    and the quantity attributable to the PLANTED weakness is the excess over it.
    The two cells are run on the same deal bank but against different opponents,
    so they are not paired and the intervals are combined in quadrature -- which
    is conservative relative to a paired design and is stated as such.
    """
    rows = load_jsonl(os.path.join(d, 'C1-floor.jsonl'))
    if not rows:
        return "*(C1-floor.jsonl absent)*\n", {}
    dtrue, resp = {}, {}
    for r in rows:
        if r.get('row') == 'dTrue':
            dtrue[r['tag']] = pts(r['winRateA'])
        elif r.get('row') in ('C1', 'C2', 'C3', 'C5'):
            resp.setdefault(r['row'], {}).setdefault(r['tag'], []).append(r)
    out = ["| class | rung | dTrue | dFound | dFound − dFound(`none`) | 95% (quadrature) | excludes 0 |",
           "|---|---|---:|---:|---:|---|:--:|"]
    scal = {}
    for cls in ('C1', 'C2', 'C3', 'C5'):
        base = resp.get(cls, {}).get('none', [])
        for tag, rs in resp.get(cls, {}).items():
            for i, r in enumerate(rs):
                bank = r.get('bank')
                b0 = None
                for x in base:
                    if x.get('bank') == bank:
                        b0 = x
                if tag == 'none' or b0 is None:
                    out.append(f"| {cls} | `{tag}` | {dtrue.get(tag, float('nan')):+.2f} | "
                               f"{pts(r['winRateA']):+.2f} | — | — | — |")
                    continue
                ex = pts(r['winRateA']) - pts(b0['winRateA'])
                se = math.hypot(_se_from_ci(r), _se_from_ci(b0))
                lo, hi = ex - 1.959963985 * se, ex + 1.959963985 * se
                out.append(f"| {cls} | `{tag}` | {dtrue.get(tag, float('nan')):+.2f} | "
                           f"{pts(r['winRateA']):+.2f} | {ex:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                           f"{'yes' if lo > 0 else 'no'} |")
                scal[f'excess:{cls}:{tag}:{bank}'] = ex
    return "\n".join(out) + "\n", scal

def pooled(d):
    """Both banks together, and the replication verdict.

    The project's rule is that a claim is replicated only if it holds in sign
    and size on both banks; the pooled estimate is reported ALONGSIDE the two
    banks, never instead of them, because pooling hides a disagreement and the
    per-bank rows are what show one.
    """
    rows = load_jsonl(os.path.join(d, 'C1-floor.jsonl'))
    if not rows:
        return "*(C1-floor.jsonl absent)*\n", {}
    dtrue, resp = {}, {}
    for r in rows:
        if r.get('row') == 'dTrue':
            dtrue[r['tag']] = pts(r['winRateA'])
        elif r.get('row') in ('C1', 'C2', 'C3', 'C5'):
            resp.setdefault(r['row'], {}).setdefault(r['tag'], []).append(r)
    out = ["| class | rung | dTrue | bank A | bank B | pooled | 95% (pooled) | n (games) | same sign |",
           "|---|---|---:|---:|---:|---:|---|---:|:--:|"]
    scal = {}
    for cls in ('C1', 'C2', 'C3', 'C5'):
        for tag, rs in resp.get(cls, {}).items():
            if len(rs) < 2:
                if rs:
                    r = rs[0]
                    out.append(f"| {cls} | `{tag}` | {dtrue.get(tag, float('nan')):+.2f} | "
                               f"{pts(r['winRateA']):+.2f} | — | — | — | {r['games']} | — |")
                continue
            a2, b2 = rs[0], rs[1]
            na, nb = a2['games'], b2['games']
            p = (pts(a2['winRateA']) * na + pts(b2['winRateA']) * nb) / (na + nb)
            sa, sb = _se_from_ci(a2), _se_from_ci(b2)
            se = math.sqrt((sa * na) ** 2 + (sb * nb) ** 2) / (na + nb)
            lo, hi = p - 1.959963985 * se, p + 1.959963985 * se
            same = (pts(a2['winRateA']) > 0) == (pts(b2['winRateA']) > 0)
            out.append(f"| {cls} | `{tag}` | {dtrue.get(tag, float('nan')):+.2f} | "
                       f"{pts(a2['winRateA']):+.2f} | {pts(b2['winRateA']):+.2f} | {p:+.2f} | "
                       f"[{lo:+.2f}, {hi:+.2f}] | {na + nb} | {'yes' if same else '**no**'} |")
            scal[f'pooled:{cls}:{tag}'] = p
            scal[f'pooledLo:{cls}:{tag}'] = lo
    return "\n".join(out) + "\n", scal

def floor_summary(d):
    """The headline the phase-1 exit criterion turns on.

    A class's DETECTION FLOOR is the smallest planted edge at which its EXCESS
    OVER THE UNHANDICAPPED CONTROL excludes zero on BOTH evaluation banks.

    Two corrections are built into that sentence and both matter.  It is the
    EXCESS, not the raw edge, because the control rung is not zero: a properly
    specified responder reaches +0.76 to +1.86 points against the unhandicapped
    incumbent, so a class that clears zero on a handicapped rung may simply be
    showing the same exploitability it shows everywhere.  And it is BOTH BANKS,
    because the project's standing rule is that a claim is replicated only if it
    holds in sign and size on both; one bank is a result about one bank.

    `excessVsRef` is the excess minus dTrue -- what the class finds beyond what
    the reference exploiter (the unhandicapped incumbent) finds on the same
    rung.  A class whose excess merely tracks dTrue is recovering the target's
    lost STRENGTH, not exploiting the planted weakness specifically.
    """
    rows = load_jsonl(os.path.join(d, 'C1-floor.jsonl'))
    if not rows:
        return "*(C1-floor.jsonl absent)*\n", {}
    dtrue, resp = {}, {}
    for r in rows:
        if r.get('row') == 'dTrue':
            dtrue[r['tag']] = pts(r['winRateA'])
        elif r.get('row') in ('C1', 'C2', 'C3', 'C5'):
            resp.setdefault(r['row'], {}).setdefault(r['tag'], []).append(r)
    out = ["| class | rungs detected (excess over control excludes 0 on both banks) | detection floor | rungs measured |",
           "|---|---|---:|---:|"]
    scal = {}
    detail = ["", "Per rung, pooled over both banks:", "",
              "| class | rung | dTrue | pooled edge | excess over control | excess − dTrue | detected |",
              "|---|---|---:|---:|---:|---:|:--:|"]
    for cls in ('C1', 'C2', 'C3', 'C5'):
        base = resp.get(cls, {}).get('none', [])
        if not base:
            continue
        found, n = [], 0
        for tag, rs in resp[cls].items():
            if tag == 'none':
                continue
            n += 1
            ok, exs = True, []
            for r in rs:
                b0 = next((x for x in base if x.get('bank') == r.get('bank')), None)
                if b0 is None:
                    ok = False; continue
                ex = pts(r['winRateA']) - pts(b0['winRateA'])
                se = math.hypot(_se_from_ci(r), _se_from_ci(b0))
                exs.append(ex)
                if ex - 1.959963985 * se <= 0:
                    ok = False
            if len(rs) < 2:
                ok = False
            dt = dtrue.get(tag, float('nan'))
            pooled_edge = sum(pts(r['winRateA']) * r['games'] for r in rs) / sum(r['games'] for r in rs)
            pooled_ctrl = sum(pts(x['winRateA']) * x['games'] for x in base) / sum(x['games'] for x in base)
            ex_pool = pooled_edge - pooled_ctrl
            detail.append(f"| {cls} | `{tag}` | {dt:+.2f} | {pooled_edge:+.2f} | {ex_pool:+.2f} | "
                          f"{ex_pool - dt:+.2f} | {'**yes**' if ok else 'no'} |")
            scal[f'excessPooled:{cls}:{tag}'] = ex_pool
            if ok:
                found.append((dt, tag))
        found.sort()
        floor_v = found[0][0] if found else float('nan')
        names = ", ".join(f"`{t}` ({v:+.2f})" for v, t in found) or "**none**"
        out.append(f"| {cls} | {names} | "
                   f"{('%.2f pts' % floor_v) if found else '**not reached**'} | {n} |")
        scal[f'floor:{cls}'] = floor_v
    return "\n".join(out) + "\n" + "\n".join(detail) + "\n", scal

# ----------------------------------------------------------------- T2 / W1 / D1
def searchstrength(d):
    rows = load_jsonl(os.path.join(d, 'T2-searchstrength.jsonl'))
    if not rows:
        return "*(T2-searchstrength.jsonl absent)*\n", {}
    out = ["| search configuration | bank | win rate vs `v06` | 95% CI | n (games) | 98/√N | games/s |",
           "|---|---:|---:|---|---:|---:|---:|"]
    scal = {}
    for r in rows:
        n = r['games']
        out.append(f"| `{r['a']}` | {r['bank']} | {100*r['winRateA']:.2f}% | "
                   f"[{100*r['ci'][0]:.2f}, {100*r['ci'][1]:.2f}] | {n} | {hw(n):.2f} | "
                   f"{r.get('gamesPerSec', float('nan')):.2f} |")
        scal[f"T2:{r['a']}:{r['bank']}"] = 100 * r['winRateA']
    return "\n".join(out) + "\n", scal

def inversion(d):
    rows = load_jsonl(os.path.join(d, 'W1-inversion.jsonl'))
    if not rows:
        return "*(W1-inversion.jsonl absent)*\n", {}
    out = ["| observer | target | model | bank | bits/ask | SE | surviving frac. | nats base → inv. | argmax base → inv. | inverted asks |",
           "|---|---|---|---:|---:|---:|---:|---|---|---:|"]
    scal = {}
    for r in rows:
        out.append(f"| `{r['a']}` | `{r['b']}` | `{r['model']}` | {r['seed']} | {r['bitsPerAsk']:.4f} | "
                   f"{r['bitsSE']:.4f} | {r['meanConsistentFrac']:.4f} | "
                   f"{r['natsBase']:.5f} → {r['natsInv']:.5f} | "
                   f"{r['argmaxBase']:.4f} → {r['argmaxInv']:.4f} | {r['inverted']} |")
        scal[f"W1:{r['b']}:{r['seed']}:bits"] = r['bitsPerAsk']
        scal[f"W1:{r['b']}:{r['seed']}:dArgmax"] = 100 * (r['argmaxInv'] - r['argmaxBase'])
    return "\n".join(out) + "\n", scal

def decisions(d):
    rows = load_jsonl(os.path.join(d, 'D1-decisions.jsonl'))
    if not rows:
        return "*(D1-decisions.jsonl absent)*\n", {}
    names = list(rows[0]['metrics'].keys())
    out = ["| arm | bank | " + " | ".join(f"`{n}`" for n in names) + " |",
           "|---|---:|" + "---:|" * len(names)]
    scal = {}
    for r in rows:
        cells = []
        for n in names:
            m = r['metrics'][n]
            cells.append(f"{m['rate']:.5f}" if m['n'] else "—")
            scal[f"D1:{r['a']}:{r['seed']}:{n}"] = m['rate']
        out.append(f"| `{r['a']}` | {r['seed']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Decision counts and intervals, first row only:")
    out.append("")
    out.append("| metric | rate | 95% CI (deal-clustered) | decisions | 98/√n |")
    out.append("|---|---:|---|---:|---:|")
    for n in names:
        m = rows[0]['metrics'][n]
        out.append(f"| `{n}` | {m['rate']:.5f} | [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}] | "
                   f"{m['n']:.0f} | {m['halfWidth98']:.3f} |")
    return "\n".join(out) + "\n", scal

def budget(d):
    rows = load_jsonl(os.path.join(d, 'C2-budget.jsonl'))
    if not rows:
        return "*(C2-budget.jsonl absent)*\n", {}
    specs = {(r['target'], r['gens'], r['pop'], r['deals']): r for r in rows if r.get('row') == 'budgetSpec'}
    out = ["| target | gens × pop × deals | games spent fitting | bank | dFound (pts) | 95% CI |",
           "|---|---|---:|---:|---:|---|"]
    scal = {}
    for r in rows:
        if r.get('row') != 'budget':
            continue
        key = (r['target'], r['gens'], r['pop'], r['fitDeals'])
        sp = specs.get(key)
        out.append(f"| `{r['target']}` | {r['gens']} × {r['pop']} × {r['fitDeals']} | "
                   f"{sp['gamesInFit'] if sp else '?'} | {r['bank']} | {pts(r['winRateA']):+.2f} | {fmt_ci(r)} |")
        scal[f"budget:{r['target']}:{r['gens']}x{r['pop']}x{r['fitDeals']}:{r['bank']}"] = pts(r['winRateA'])
    return "\n".join(out) + "\n", scal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='research/v07/results')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--paper', action='store_true',
                    help='emit paper/numbers_v07_generated.tex and paper/tables_v07/*.tex '
                         'instead of the INSTRUMENT.md markdown')
    a = ap.parse_args()
    if a.paper:
        return paper_main()
    scal = {}
    sections = [("Throughput (T1)", throughput),
                ("Detection floor summary", floor_summary),
                ("Detection floor (C1)", floor),
                ("Both banks pooled", pooled),
                ("Edge above the unhandicapped control", excess),
                ("Fast-search strength (T2)", searchstrength), ("Budget curve (C2)", budget),
                ("Transcript inversion (W1)", inversion), ("Per-decision channel (D1)", decisions)]
    body = []
    for title, fn in sections:
        txt, s = fn(a.dir)
        scal.update(s)
        body.append(f"\n## {title}\n\n{txt}")
    if a.json:
        print(json.dumps(scal, indent=1, sort_keys=True))
    else:
        print("".join(body))



# =============================================================================
# --paper : emit paper/numbers_v07_generated.tex and paper/tables_v07/*.tex
# =============================================================================
# The v0.6 manuscript's convention, kept exactly: every macro is written as
# \providecommand followed by \renewcommand so this generated file always wins
# over the placeholder in numbers_v07.tex, and every macro carries a comment
# naming the artifact and field it was read from.  paper/check_provenance.py
# reads those comments.  No number in sections_v07/ is typed by hand.
import hashlib, re as _re, time as _time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PAPER = os.path.join(_ROOT, 'paper')
_TBL = os.path.join(_PAPER, 'tables_v07')
_RES = os.path.join(_ROOT, 'research', 'v07', 'results')

OUT = []
_SEEN = {}

_DIGITWORD = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
              '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

def _mac(name):
    """A LaTeX control word is letters only: \\vsevenBtwoFcheap is one macro,
    \\vsevenB2Fcheap is \\vsevenB followed by the text "2Fcheap".  Any digit that
    reaches a macro name is therefore a silent typesetting bug, so digits are
    spelled and the result is asserted alphabetic."""
    out = ''.join(_DIGITWORD[c] if c.isdigit() else c for c in str(name))
    if not out.isalpha():
        raise SystemExit('macro name is not alphabetic after spelling digits: %r -> %r'
                         % (name, out))
    return out

def emit(name, value, src):
    """One generated macro.  Re-emitting a name with a different value is a bug
    in this script, not something to paper over, so it raises."""
    name = _mac(name)
    value = str(value)
    if name in _SEEN and _SEEN[name] != value:
        raise SystemExit('macro %s emitted twice with different values: %r vs %r'
                         % (name, _SEEN[name], value))
    _SEEN[name] = value
    # A signed number reaching running prose as ASCII typesets with a text hyphen
    # and a text plus, while the same figure inside a table cell or a confidence
    # interval is set in math and gets a proper minus.  The two were appearing in
    # the same sentence.  \ensuremath settles it, and it also makes the sign
    # unbreakable: -1.75 was splitting across a line break as "and -" / "1.75".
    out = ('\\ensuremath{%s}' % value) if _re.fullmatch(r'[+-][0-9][0-9.,]*', value) else value
    OUT.append('%% %s\n\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}' % (src, name, name, out))

def _n(x, d=2):
    return ('%.*f' % (d, x))

def _sg(x, d=2):
    return ('%+.*f' % (d, x))

_NUMWORD = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
            7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
            13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
            17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty'}

def _word(n):
    """House style spells small counts in running prose and prints numerals in
    tables, so a count used in both places needs both forms generated."""
    n = int(n)
    return _NUMWORD.get(n, str(n))

def _commafy(n):
    return '{:,}'.format(int(n))

def _tex(s):
    """Escape a spec string for LaTeX text.  Specs carry _ and & and %."""
    t = (str(s).replace('\\', '\\textbackslash ').replace('_', '\\_')
         .replace('%', '\\%').replace('&', '\\&').replace('#', '\\#'))
    # A Markdown code span read out of an artifact would otherwise typeset as
    # two opening quotes, `like this`, because LaTeX has no closing backtick.
    return _re.sub(r'`([^`]*)`', r'\\texttt{\1}', t)

BANK1, BANK2, BANK3 = 7090001, 7090002, 7090003

def _rows(fn):
    rs = load_jsonl(os.path.join(_RES, fn))
    return [r for r in rs if r.get('ok')]

def _edge(r):
    return pts(r['match']['winRateA'])

def _ci(r):
    return pts(r['match']['ci'][0]), pts(r['match']['ci'][1])

def _se(r):
    lo, hi = _ci(r)
    return (hi - lo) / (2 * 1.96)

def pool(rs):
    """Two banks of equal size: mean of the edges, se = sqrt(se1^2+se2^2)/2.
    FINAL-RESULTS.md section 0 fixes this arithmetic; it is repeated here rather
    than imported because this script must be readable on its own."""
    es = [_edge(r) for r in rs]
    ss = [_se(r) for r in rs]
    m = sum(es) / len(es)
    sp = math.sqrt(sum(s * s for s in ss)) / len(ss)
    return m, m - 1.96 * sp, m + 1.96 * sp, sp

def delta(a, b):
    """A difference of two POOLED cells, half-widths combined in quadrature.
    PREREGISTRATION 5.2 fixes this and calls it conservative: the arms are
    paired on deals but the harness gives no paired delta across two cells."""
    ma, _, _, sa = pool(a)
    mb, _, _, sb = pool(b)
    d = ma - mb
    h = 1.96 * math.sqrt(sa * sa + sb * sb)
    return d, d - h, d + h

def replicates(rs):
    es = [_edge(r) for r in rs]
    return all(e > 0 for e in es) or all(e < 0 for e in es)

# Display names.  Harness identifiers (FROZEN, INCUMBENT) are how the artifacts
# label the arms; the paper names them consistently as the agents they are, and
# uses the harness spelling only where a command has to be reproduced.
DISPLAY = {'FROZEN': 'SESTINA v1.0', 'INCUMBENT': 'v0.6 deployed', 'v06': 'v0.6',
           'F-cheap': '\\texttt{F-cheap}', 'F-mid': '\\texttt{F-mid}',
           'composite': 'phase-2 composite', 'v05': 'v0.5', 'v04': 'v0.4',
           'v03': 'v0.3', 'v02': 'v0.2', 'F-fast': '\\texttt{F-fast}',
           'itself': 'itself'}

def disp(name):
    # A name not in DISPLAY is a raw configuration identifier -- a panel member,
    # a scripted archetype, a sealed adversary.  The prose sets those in
    # typewriter, so the tables must too, or SEALED:X01xC3f ends up roman in one
    # column and \texttt{F-cheap} typewriter in the cell below it.
    return DISPLAY.get(name, '\\texttt{%s}' % _tex(name))

TABLE_SPEC = {}

def writeTable(fn, lines, spec=None, header=None, longtable=False, xtab=False):
    """Emit one table body.

    `xtab` selects the X-column form: `tabularx` at \\linewidth, or `xltabular`
    when `longtable` is also set.  Nothing this script emits may be wider than
    the text block, and the wide tables get there by wrapping text in X columns
    rather than by shrinking the type or reaching into the margin.
    """
    os.makedirs(_TBL, exist_ok=True)
    spec, header = (spec, header) if spec else TABLE_SPEC[fn]
    with open(os.path.join(_TBL, fn), 'w') as f:
        if longtable and xtab:
            f.write('\\begin{xltabular}{\\linewidth}{%s}\n\\toprule\n%s \\\\\n\\midrule\n'
                    '\\endfirsthead\n\\toprule\n%s \\\\\n\\midrule\n\\endhead\n'
                    '\\midrule\n\\multicolumn{%d}{r@{}}{\\footnotesize\\itshape continued overleaf}\\\\\n'
                    '\\endfoot\n\\bottomrule\n\\endlastfoot\n'
                    % (spec, header, header, header.count('&') + 1))
            f.write('\n'.join(lines) + '\n\\end{xltabular}\n')
        elif longtable:
            f.write('\\begin{longtable}{%s}\n\\toprule\n%s \\\\\n\\midrule\n\\endfirsthead\n'
                    '\\toprule\n%s \\\\\n\\midrule\n\\endhead\n' % (spec, header, header))
            f.write('\n'.join(lines) + '\n\\bottomrule\n\\end{longtable}\n')
        elif xtab:
            f.write('\\begin{tabularx}{\\linewidth}{%s}\n\\toprule\n%s \\\\\n\\midrule\n'
                    % (spec, header))
            f.write('\n'.join(lines) + '\n\\bottomrule\n\\end{tabularx}\n')
        else:
            f.write('\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n' % (spec, header))
            f.write('\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')


# ---------------------------------------------------------------- B3, the headline
# The project's standing reporting rule (paper/sections_v06/10-protocol.tex,
# sec:protocol-metrics) is that the worst case leads and an aggregate never
# does.  So B3 is emitted first and the panel mean is emitted last, labelled.
ARMTAG = {'FROZEN': 'Frozen', 'INCUMBENT': 'Incumbent', 'F-cheap': 'Fcheap',
          'composite': 'Composite'}
# A panel member that IS the arm gives that arm a free 0.00 cell, which can only
# flatter its worst case.  FROZEN is not a panel member and carries none, so the
# worst case is also computed with self-cells excluded and both are printed.
SELFCELL = {'INCUMBENT': ('F-fast', 'SEALED:S-reference-0'),
            'F-cheap': ('F-cheap',), 'composite': ('composite',), 'FROZEN': ()}

def b3_panel_table():
    rows = _rows('P5-B3.jsonl')
    arms = ['FROZEN', 'INCUMBENT', 'F-cheap', 'composite']
    panel, cls, deals = [], {}, {}
    for r in rows:
        if r['arm'] == 'FROZEN' and r['bank'] == BANK1:
            panel.append(r['opp']); cls[r['opp']] = r['cls']; deals[r['opp']] = r['deals']
    cell = {}
    for a in arms:
        for p in panel:
            rs = [r for r in rows if r['arm'] == a and r['opp'] == p]
            if len(rs) == 2:
                cell[(a, p)] = pool(rs) + (tuple(sorted((r['bank'], _edge(r)) for r in rs)),)
    emit('vsevenPanelMembers', str(len(panel)), 'P5-B3.jsonl: named panel members')
    # Three of the named members are the same policy under a sealed alias, so the
    # panel MEAN counts three policies twice. Minimax regret and the worst cell
    # are unaffected; a weighted mean is not, and the denominator should say so.
    specs = {}
    for p_ in panel:
        r0 = [r for r in rows if r['arm'] == 'FROZEN' and r['opp'] == p_ and r['bank'] == BANK1]
        if r0:
            specs[p_] = r0[0].get('oppSpec', p_)
    emit('vsevenPanelDistinct', str(len(set(specs.values()))),
         'P5-B3.jsonl: DISTINCT policies among the named panel members')
    emit('vsevenPanelCells', str(len(rows)), 'P5-B3.jsonl: scored cells, 4 arms x 31 members x 2 banks')
    emit('vsevenPanelNear', str(sum(1 for p in panel if cls[p] == 'near')), 'P5-B3.jsonl: near-class members')
    emit('vsevenPanelFar', str(sum(1 for p in panel if cls[p] == 'far')), 'P5-B3.jsonl: far-class members')
    emit('vsevenPanelExpensive', str(sum(1 for p in panel if cls[p] == 'expensive')), 'P5-B3.jsonl: expensive-class members')
    for c, tag in (('near', 'Near'), ('far', 'Far'), ('expensive', 'Expensive')):
        d = {deals[p] for p in panel if cls[p] == c}
        emit('vsevenDeals%s' % tag, _commafy(sorted(d)[0]), 'P5-B3.jsonl: deals per bank, %s class' % c)

    # ---- worst case per arm, with and without the arm's own free self-cell
    worst_lines = []
    for a in arms:
        vals = [(cell[(a, p)][0], p) for p in panel if (a, p) in cell]
        w, wp = min(vals)
        _, lo, hi, _ = cell[(a, wp)][:4]
        per = cell[(a, wp)][4]
        t = ARMTAG[a]
        emit('vsevenWorst%s' % t, _sg(w), 'P5-B3.jsonl: %s worst pooled panel cell' % a)
        emit('vsevenWorst%sLo' % t, _sg(lo), 'P5-B3.jsonl: %s worst cell, pooled lower bound' % a)
        emit('vsevenWorst%sHi' % t, _sg(hi), 'P5-B3.jsonl: %s worst cell, pooled upper bound' % a)
        emit('vsevenWorst%sOpp' % t, _tex(wp), 'P5-B3.jsonl: %s worst cell opponent' % a)
        # Whether the WORST CELL replicates is a separate question from whether
        # the arm's edge on it does, and the report's own rule (a claim whose
        # sign does not agree on both banks is NOT REPLICATED) applies to it.
        # For an arm whose worst cell straddles zero the attaining member can
        # differ by bank too, so both are emitted.
        perbank = {}
        for bank in (BANK1, BANK2):
            cand = []
            for p in panel:
                rs = [r for r in rows if r['arm'] == a and r['opp'] == p and r['bank'] == bank]
                if rs:
                    cand.append((_edge(rs[0]), p))
            if cand:
                perbank[bank] = min(cand)
        if len(perbank) == 2:
            (e1, p1), (e2, p2) = perbank[BANK1], perbank[BANK2]
            emit('vsevenWorst%sBankOne' % t, _sg(e1), 'P5-B3.jsonl: %s worst cell on bank 7090001' % a)
            emit('vsevenWorst%sBankTwo' % t, _sg(e2), 'P5-B3.jsonl: %s worst cell on bank 7090002' % a)
            emit('vsevenWorst%sBankOneOpp' % t, _tex(p1), 'P5-B3.jsonl: %s worst-cell member on bank 7090001' % a)
            emit('vsevenWorst%sBankTwoOpp' % t, _tex(p2), 'P5-B3.jsonl: %s worst-cell member on bank 7090002' % a)
            emit('vsevenWorst%sRepl' % t, 'yes' if (e1 > 0) == (e2 > 0) else 'NO',
                 'P5-B3.jsonl: %s worst cell, sign agrees on both banks' % a)
            emit('vsevenWorst%sSameOpp' % t, 'yes' if p1 == p2 else 'NO',
                 'P5-B3.jsonl: %s worst cell attained at the same member on both banks' % a)
        nonself = [(v, p) for v, p in vals if p not in SELFCELL[a]]
        w2, wp2 = min(nonself)
        emit('vsevenWorst%sExSelf' % t, _sg(w2), 'P5-B3.jsonl: %s worst cell excluding its own self-cells' % a)
        emit('vsevenWorst%sExSelfOpp' % t, _tex(wp2), 'P5-B3.jsonl: %s worst non-self cell opponent' % a)
        worst_lines.append('%s & %s & $%s$ & $[%s, %s]$ & $%s$ / $%s$ \\\\' %
                           (disp(a), disp(wp), _sg(w), _sg(lo), _sg(hi),
                            _sg(per[0][1]), _sg(per[1][1])))
    writeTable('worstcase.tex', worst_lines, 'llrcr',
               'Configuration & Worst cell & Edge (pp) & 95\\% CI & Per bank')

    # ---- minimax regret: for each member, the best arm on it; an arm's regret
    # is its largest shortfall from that best, maximised over members.
    reg_lines = []
    regret = {}
    for a in arms:
        best_shortfall, at = -1e9, None
        runner = None
        for p in panel:
            if (a, p) not in cell: continue
            best = max(cell[(x, p)][0] for x in arms if (x, p) in cell)
            s = best - cell[(a, p)][0]
            if s > best_shortfall:
                runner, best_shortfall, at = at, s, p
        regret[a] = (best_shortfall, at)
        t = ARMTAG[a]
        emit('vsevenRegret%s' % t, _n(best_shortfall), 'P5-B3.jsonl: %s minimax regret over the shared panel' % a)
        emit('vsevenRegret%sOpp' % t, _tex(at), 'P5-B3.jsonl: member at which %s attains its regret' % a)
        reg_lines.append('%s & %s & %s \\\\' % (disp(a), _n(best_shortfall), disp(at)))
    # how close the incumbent's attaining member is to its runner-up: the regret
    # VALUE is stable and the name attached to it is not, and that is only worth
    # saying if the margin is printed.
    sh = sorted((max(cell[(x, p)][0] for x in arms if (x, p) in cell) - cell[('INCUMBENT', p)][0])
                for p in panel if ('INCUMBENT', p) in cell)
    emit('vsevenRegretIncumbentGap', _n(sh[-1] - sh[-2], 3),
         'P5-B3.jsonl: INCUMBENT regret, attaining member minus runner-up')
    order = sorted(arms, key=lambda a: regret[a][0])
    emit('vsevenRegretBest', _tex(order[0]), 'P5-B3.jsonl: arm with the lowest minimax regret')
    emit('vsevenRegretFrozenRank', {a: i + 1 for i, a in enumerate(order)}['FROZEN'],
         'P5-B3.jsonl: FROZEN rank of four on minimax regret, 1 = best')
    TABLE_SPEC['regret.tex'] = ('lrr', 'Configuration & Minimax regret (pp) & Attained against')
    writeTable('regret.tex', ['%s & %s & %s \\\\' % (disp(a), _n(regret[a][0]), disp(regret[a][1]))
                              for a in order])

    # the far archetypes, where the regret is concentrated
    far_lines = []
    for p in [x for x in panel if cls[x] == 'far']:
        far_lines.append('%s & %s \\\\' % (disp(p), ' & '.join(
            _sg(cell[(a, p)][0]) for a in arms if (a, p) in cell)))
    fars = [cell[(a, p)][0] for p in panel if cls[p] == 'far' for a in arms if (a, p) in cell]
    emit('vsevenFarSpanLo', _n(min(fars)), 'P5-B3.jsonl: smallest edge any arm takes on a far archetype')
    emit('vsevenFarSpanHi', _n(max(fars)), 'P5-B3.jsonl: largest edge any arm takes on a far archetype')
    TABLE_SPEC['farpanel.tex'] = ('lrrrr', 'Far archetype & SESTINA v1.0 & v0.6 & \\texttt{F-cheap} & Composite')
    writeTable('farpanel.tex', far_lines)

    # the hardest sealed adversary, where the ordering reverses
    hard = min(((cell[('INCUMBENT', p)][0], p) for p in panel if p.startswith('SEALED')))
    hp = hard[1]
    emit('vsevenHardSealed', _tex(hp), 'P5-B3.jsonl: sealed member on which INCUMBENT is worst')
    beat = [a for a in arms if cell[(a, hp)][1] > 0]
    emit('vsevenHardBeaters', str(len(beat)),
         'P5-B3.jsonl: arms beating that member with a lower bound above zero')
    emit('vsevenHardBeaterNames', ', '.join(_tex(a) for a in beat),
         'P5-B3.jsonl: which arms they are')
    for a in arms:
        m, lo, hi, _ = cell[(a, hp)][:4]
        emit('vsevenHard%s' % ARMTAG[a], _sg(m), 'P5-B3.jsonl: %s against %s' % (a, hp))
        emit('vsevenHard%sLo' % ARMTAG[a], _sg(lo), 'P5-B3.jsonl: %s against %s, lower bound' % (a, hp))
        emit('vsevenHard%sHi' % ARMTAG[a], _sg(hi), 'P5-B3.jsonl: %s against %s, upper bound' % (a, hp))

    # ---- the aggregate, emitted LAST and named as a diagnostic
    for a in arms:
        mean = sum(cell[(a, p)][0] for p in panel if (a, p) in cell) / len(panel)
        emit('vsevenPanelMean%s' % ARMTAG[a], _sg(mean),
             'P5-B3.jsonl: %s mean over the 31 shared cells -- DIAGNOSTIC, never the headline' % a)

    # ---- the full 31 x 4 table, so a reader can take the worst cell themselves
    full = []
    for p in panel:
        full.append('%s & %s & %s \\\\' % (disp(p), cls[p], ' & '.join(
            ('$%s$' % _sg(cell[(a, p)][0])) if (a, p) in cell else '---' for a in arms)))
    writeTable('panelfull.tex', full, '@{}l l r r r r@{}',
               'Panel member & Class & SESTINA v1.0 & v0.6 & \\texttt{F-cheap} & Composite',
               longtable=True)
    return cell, panel, cls


# ---------------------------------------------------------------- B2, the frontier
B2LAB = {'B2.1': ('Incumbent', 'v06'), 'B2.2': ('Fcheap', 'F-cheap'),
         'B2.3': ('Fmid', 'F-mid'), 'B2.4': ('Composite', 'the phase-2 composite'),
         'B2.5': ('Vfive', 'v05')}

def b2_frontier():
    rows = _rows('P5-B2.jsonl')
    lines = []
    for cellid in sorted({r['cell'] for r in rows}):
        rs = [r for r in rows if r['cell'] == cellid]
        m, lo, hi, _ = pool(rs)
        tag, name = B2LAB[cellid]
        per = {r['bank']: _edge(r) for r in rs}
        n = sum(r['match']['games'] for r in rs)
        emit('vsevenBtwo%s' % tag, _sg(m), 'P5-B2.jsonl %s: FROZEN vs %s, pooled edge' % (cellid, name))
        emit('vsevenBtwo%sLo' % tag, _sg(lo), 'P5-B2.jsonl %s: pooled lower bound' % cellid)
        emit('vsevenBtwo%sHi' % tag, _sg(hi), 'P5-B2.jsonl %s: pooled upper bound' % cellid)
        emit('vsevenBtwo%sN' % tag, _commafy(n), 'P5-B2.jsonl %s: games' % cellid)
        emit('vsevenBtwo%sBankOne' % tag, _sg(per[BANK1]), 'P5-B2.jsonl %s: bank 7090001' % cellid)
        emit('vsevenBtwo%sBankTwo' % tag, _sg(per[BANK2]), 'P5-B2.jsonl %s: bank 7090002' % cellid)
        emit('vsevenBtwo%sMag' % tag, _n(abs(m)),
             'P5-B2.jsonl %s: pooled edge, unsigned, for prose that supplies its own direction' % cellid)
        emit('vsevenBtwo%sRepl' % tag, 'yes' if replicates(rs) else 'no',
             'P5-B2.jsonl %s: sign agrees on both banks' % cellid)
        lines.append('%s & %s & $%s$ & $[%s, %s]$ & %s & %s \\\\' %
                     (cellid, disp(name) if name in DISPLAY else _tex(name), _sg(m), _sg(lo), _sg(hi),
                      _commafy(n), 'yes' if replicates(rs) else 'no'))
        # declaration accuracy is emitted as a bare rate: match --json gives it no
        # interval of any kind, which is what PREREGISTRATION 3 correction 4 and
        # deviation D11 turn on.  It is a diagnostic and carries no claim.
        emit('vsevenBtwo%sDeclA' % tag, _n(sum(r['match']['declAccA'] for r in rs) / len(rs), 3),
             'P5-B2.jsonl %s: FROZEN declaration accuracy, a bare rate with no interval' % cellid)
    TABLE_SPEC['frontier.tex'] = ('llrrrr', 'Cell & Opponent & Edge (pp) & 95\\% CI & Games & Sign replicates')
    writeTable('frontier.tex', lines)
    fc = [r for r in rows if r['cell'] == 'B2.2']
    emit('vsevenBtwoFcheapHalfWidth', _n(pool(fc)[3] * 1.96),
         'P5-B2.jsonl B2.2: pooled half-width of the primary claim, in points')
    das = [r['match']['declAccA'] for r in rows]
    dbs = [r['match']['declAccB'] for r in rows]
    emit('vsevenDeclFrozenLo', _n(min(das), 3), 'P5-B2.jsonl: lowest FROZEN declaration accuracy over the five cells')
    emit('vsevenDeclFrozenHi', _n(max(das), 3), 'P5-B2.jsonl: highest FROZEN declaration accuracy')
    emit('vsevenDeclOppLo', _n(min(dbs), 3), 'P5-B2.jsonl: lowest opponent declaration accuracy')
    emit('vsevenDeclOppHi', _n(max(dbs), 3), 'P5-B2.jsonl: highest opponent declaration accuracy')


# ---------------------------------------------------------------- B4, fresh adversaries
def b4_adversaries():
    fits = {r['id']: r for r in load_jsonl(os.path.join(_RES, 'P5-B4fits.jsonl'))
            if r.get('battery') == 'B4fit'}
    rows = _rows('P5-B4eval.jsonl')
    ZTAG = {'Z01': 'Zone', 'Z02': 'Ztwo', 'Z03': 'Zthree', 'Z04': 'Zfour',
            'Z05': 'Zfive', 'Z06': 'Zsix', 'Z07': 'Zseven', 'Z08': 'Zeight'}
    lines, uppers = [], []
    for zid in sorted({r['cell'] for r in rows}):
        rs = [r for r in rows if r['cell'] == zid]
        # the row is the ADVERSARY's edge over FROZEN: the adversary is arm A.
        m, lo, hi, _ = pool(rs)
        per = {r['bank']: _edge(r) for r in rs}
        uppers.append((hi, zid))
        emit('vseven%sEdge' % ZTAG[zid], _sg(m), 'P5-B4eval.jsonl %s: fitted adversary edge over FROZEN' % zid)
        emit('vseven%sLo' % ZTAG[zid], _sg(lo), 'P5-B4eval.jsonl %s: lower bound' % zid)
        emit('vseven%sHi' % ZTAG[zid], _sg(hi), 'P5-B4eval.jsonl %s: upper bound' % zid)
        emit('vseven%sKpi' % ZTAG[zid], _tex(rs[0]['kpi']), 'P5-B4eval.jsonl %s: search objective' % zid)
        emit('vseven%sCls' % ZTAG[zid], _tex(rs[0]['cls']), 'P5-B4eval.jsonl %s: adversary class' % zid)
        lines.append('%s & %s & %s & $%s$ & $[%s, %s]$ & $%s$ / $%s$ & %s \\\\' %
                     (zid, _tex(rs[0]['cls']), _tex(rs[0]['kpi']), _sg(m), _sg(lo), _sg(hi),
                      _sg(per[BANK1]), _sg(per[BANK2]),
                      'yes' if replicates(rs) else 'no'))
    writeTable('adversaries.tex', lines, '@{}l L l r c r c@{}',
               'ID & Class & Objective & Edge (pp) & 95\\% CI & Per bank & Repl.', xtab=True)
    for f in fits.values():
        a = f.get('argv') or []
        for flag, tag in (('--gens=', 'Gens'), ('--pop=', 'Pop'), ('--games=', 'Deals')):
            v = [x[len(flag):] for x in a if x.startswith(flag)]
            if v:
                emit('vsevenBfour%s' % tag, v[0], 'P5-B4fits.jsonl: %s in the search argv' % flag.strip('-='))
        break
    # The fitting-to-holdout gap needs the fitting side, and it is in the trace
    # rather than in any prose document.  bestScore is a paired margin in win-rate
    # units; the final generation is the one whose vector was evaluated.
    z1 = load_jsonl(os.path.join(_RES, 'P5-Z01.jsonl'))
    gens = [r for r in z1 if 'bestScore' in r]
    if gens:
        emit('vsevenZoneFitMargin', _sg(100.0 * gens[-1]['bestScore']),
             'P5-Z01.jsonl: paired margin of the final generation, on its own fitting stream')
        emit('vsevenZoneFitPeak', _sg(100.0 * max(g['bestScore'] for g in gens)),
             'P5-Z01.jsonl: best paired margin any generation reached on its fitting stream')
    hi, zid = max(uppers)
    emit('vsevenBfourArms', str(len(uppers)), 'P5-B4eval.jsonl: independent adversary searches evaluated')
    emit('vsevenBfourArmsWord', _word(len(uppers)),
         'P5-B4eval.jsonl: the same count spelled, for running prose')
    emit('vsevenBfourFits', str(sum(1 for f in fits.values() if f.get('ok'))),
         'P5-B4fits.jsonl: searches that produced a weight vector')
    emit('vsevenBfourWorstUpper', _sg(hi), 'P5-B4eval.jsonl: largest upper bound over all arms')
    emit('vsevenBfourWorstUpperId', ZTAG[zid], 'P5-B4eval.jsonl: arm attaining that upper bound')
    emit('vsevenBfourFloorGap', _n(1.53 - hi), 'P5-B4eval.jsonl: 1.53 floor minus the largest upper bound')
    lo_all = min(pool([r for r in rows if r['cell'] == z])[0] for z in {r['cell'] for r in rows})
    emit('vsevenBfourWorstLoss', _n(-lo_all), 'P5-B4eval.jsonl: largest loss any arm takes, in points')
    emit('vsevenBfourBestLoss', _n(-max(pool([r for r in rows if r['cell'] == z])[0]
                                        for z in {r['cell'] for r in rows})),
         'P5-B4eval.jsonl: smallest loss any arm takes, in points')


# ---------------------------------------------------------------- B5, attribution
B5TAG = {'A-search': 'Search', 'A-rtie': 'Rtie', 'A-urgoff': 'Urgoff', 'A-stall': 'Stall',
         'A-r12': 'Rtwelve', 'A-m2': 'Mtwo', 'L-search': 'Search', 'L-rtie': 'Rtie',
         'L-urgoff': 'Urgoff', 'L-stall': 'Stall', 'L-r12': 'Rtwelve'}
# (macro tag, conceptual component name, harness identifier).  The paper's
# attribution table prints the concept with the identifier in parentheses,
# so that a reader meets the same names Section 6 uses; the macro tags and
# the artifact cell names are unchanged.
COMPONENTS = [('Rtwelve', 'half-suit contestation', 'r12=25'),
              ('Search', 'guarded search', 's1=1'),
              ('Rtie', 'public-history tie-break', 'rtie=1'),
              ('Urgoff', 'urgency escalation off', 'urgency-off'),
              ('Stall', 'stall detector', 'stall=12')]

def b5_attribution():
    rows = _rows('P5-B5.jsonl')
    by = {}
    for cid in {r['cell'] for r in rows}:
        by[cid] = [r for r in rows if r['cell'] == cid]
    ref = by['REF-FROZEN']
    rm, rlo, rhi, _ = pool(ref)
    emit('vsevenBfiveWhole', _sg(rm), 'P5-B5.jsonl REF-FROZEN: FROZEN vs INCUMBENT on the lattice banks')
    emit('vsevenBfiveWholeLo', _sg(rlo), 'P5-B5.jsonl REF-FROZEN: lower bound')
    emit('vsevenBfiveWholeHi', _sg(rhi), 'P5-B5.jsonl REF-FROZEN: upper bound')
    lines, addins = [], []
    for tag, name, ident in COMPONENTS:
        a = [c for c in by if c.startswith('A-') and B5TAG[c] == tag][0]
        am, alo, ahi, _ = pool(by[a])
        emit('vsevenAddIn%s' % tag, _sg(am), 'P5-B5.jsonl %s: add-one-in from v06' % a)
        emit('vsevenAddIn%sLo' % tag, _sg(alo), 'P5-B5.jsonl %s: lower bound' % a)
        emit('vsevenAddIn%sHi' % tag, _sg(ahi), 'P5-B5.jsonl %s: upper bound' % a)
        addins.append(am)
        l = [c for c in by if c.startswith('L-') and B5TAG[c] == tag]
        if l:
            d, dlo, dhi = delta(ref, by[l[0]])
            emit('vsevenLoo%s' % tag, _sg(d), 'P5-B5.jsonl %s: leave-one-out drop from FROZEN' % l[0])
            emit('vsevenLoo%sLo' % tag, _sg(dlo), 'P5-B5.jsonl %s: lower bound' % l[0])
            emit('vsevenLoo%sHi' % tag, _sg(dhi), 'P5-B5.jsonl %s: upper bound' % l[0])
            loo = '$%s$ & $[%s, %s]$' % (_sg(d), _sg(dlo), _sg(dhi))
        else:
            loo = '--- & ---'
        lines.append('%s (\\texttt{%s}) & $%s$ & $[%s, %s]$ & %s \\\\'
                     % (_tex(name), _tex(ident), _sg(am), _sg(alo), _sg(ahi), loo))
    # m2=0 is preregistered as a B5 cell although the freeze does not carry it
    mm, mlo, mhi, _ = pool(by['A-m2'])
    emit('vsevenAddInMtwo', _sg(mm), 'P5-B5.jsonl A-m2: add-one-in, NOT carried by the freeze')
    emit('vsevenAddInMtwoLo', _sg(mlo), 'P5-B5.jsonl A-m2: lower bound')
    emit('vsevenAddInMtwoHi', _sg(mhi), 'P5-B5.jsonl A-m2: upper bound')
    lines.append('rescoring switch (\\texttt{m2=0}; not in the freeze) & $%s$ & $[%s, %s]$ & --- & --- \\\\'
                 % (_sg(mm), _sg(mlo), _sg(mhi)))
    writeTable('attribution.tex', lines, 'lrcrc',
               'Component & Add-one-in & 95\\% CI & Leave-one-out & 95\\% CI')
    naive = sum(addins)
    emit('vsevenBfiveSum', _sg(naive), 'P5-B5.jsonl: naive sum of the FIVE add-one-in cells the freeze carries')
    emit('vsevenBfiveSumSix', _sg(naive + mm), 'P5-B5.jsonl: the six-term sum including A-m2, printed for D12')
    # The ratio is whole/sum.  The sum is five independent add-one-in cells, so
    # its half-width is theirs in quadrature; the ratio's interval follows from
    # propagating both.  Without this the paper cannot say what lies inside it.
    ses = [pool(by[[c for c in by if c.startswith('A-') and B5TAG[c] == t][0]])[3]
           for t, _, _ in COMPONENTS]
    s_sum = math.sqrt(sum(x * x for x in ses))
    _, _, _, s_ref = pool(ref)
    r = rm / naive
    s_r = abs(r) * math.sqrt((s_ref / rm) ** 2 + (s_sum / naive) ** 2)
    emit('vsevenSubAdditivity', _n(r, 3), 'P5-B5.jsonl: measured whole / naive sum')
    emit('vsevenSubAdditivityLo', _n(r - 1.96 * s_r, 3), 'P5-B5.jsonl: ratio lower bound')
    emit('vsevenSubAdditivityHi', _n(r + 1.96 * s_r, 3), 'P5-B5.jsonl: ratio upper bound')
    emit('vsevenSubAdditivityOverstate', _n(naive - rm), 'P5-B5.jsonl: points the naive sum overstates by')
    # the location test of PREREGISTRATION 6 item 3, which must be read per bank
    loc = []
    for bank in (BANK1, BANK3):
        w = _edge([r for r in ref if r['bank'] == bank][0])
        drops = [(w - _edge([r for r in by[c] if r['bank'] == bank][0]), c)
                 for c in by if c.startswith('L-')]
        big, at = max(drops)
        located = big >= w / 3.0
        b = 'One' if bank == BANK1 else 'Three'
        emit('vsevenLocWhole%s' % b, _n(w, 3), 'P5-B5.jsonl: whole on bank %d' % bank)
        emit('vsevenLocThird%s' % b, _n(w / 3.0, 3), 'P5-B5.jsonl: one third of the whole on bank %d' % bank)
        emit('vsevenLocBiggest%s' % b, _sg(big, 3), 'P5-B5.jsonl: largest single leave-one-out drop on bank %d' % bank)
        emit('vsevenLocAt%s' % b, _tex(at), 'P5-B5.jsonl: component attaining it on bank %d' % bank)
        emit('vsevenLocVerdict%s' % b, 'LOCATED' if located else 'NOT LOCATED',
             'P5-B5.jsonl: PREREGISTRATION 6 item 3 verdict on bank %d' % bank)
        loc.append((w, big, located))
    emit('vsevenLocReplicates', 'yes' if loc[0][2] == loc[1][2] else 'no',
         'P5-B5.jsonl: whether the location verdict agrees across banks')
    emit('vsevenLocSplit', _n(abs(loc[0][1] - loc[1][1]), 3),
         'P5-B5.jsonl: difference between the two banks largest drops')


# ---------------------------------------------------------------- B6, partner regimes
# The partner-regime table is emitted WITH intervals on every delta.  The
# phase-5 results document compresses them out of its markdown; the artifact
# carries a clustered interval on every cell, so the paper prints them.
PTAG = {'itself': 'Itself', 'v06': 'Vsix', 'v05': 'Vfive', 'v04': 'Vfour', 'v03': 'Vthree',
        'detective': 'Detective', 'withholder': 'Withholder', 'lockout': 'Lockout'}
PORDER = ['itself', 'v06', 'v05', 'v04', 'v03', 'detective', 'withholder', 'lockout']

def b6_partners():
    rows = _rows('P5-B6.jsonl')
    out = {}
    for opp in ('v05', 'v06'):
        lines = []
        parts = [p for p in PORDER if any(r['opp'] == opp and r['partners'] == p for r in rows)]
        for p in parts:
            fr = [r for r in rows if r['opp'] == opp and r['arm'] == 'FROZEN' and r['partners'] == p]
            inc = [r for r in rows if r['opp'] == opp and r['arm'] == 'INCUMBENT' and r['partners'] == p]
            if len(fr) != 2 or len(inc) != 2: continue
            d, dlo, dhi = delta(fr, inc)
            per = {r['bank']: _edge(r) - _edge([x for x in inc if x['bank'] == r['bank']][0]) for r in fr}
            rep = (per[BANK1] > 0) == (per[BANK2] > 0) and per[BANK1] != 0 and per[BANK2] != 0
            tag = 'Opp%s%s' % ('Five' if opp == 'v05' else 'Six', PTAG[p])
            emit('vseven%s' % tag, _sg(d), 'P5-B6.jsonl: FROZEN-INCUMBENT delta, opponent %s, partners %s' % (opp, p))
            emit('vseven%sLo' % tag, _sg(dlo), 'P5-B6.jsonl: lower bound, opponent %s, partners %s' % (opp, p))
            emit('vseven%sHi' % tag, _sg(dhi), 'P5-B6.jsonl: upper bound, opponent %s, partners %s' % (opp, p))
            emit('vseven%sRepl' % tag, 'yes' if rep else 'NO',
                 'P5-B6.jsonl: sign of the delta agrees on both banks, opponent %s, partners %s' % (opp, p))
            lines.append('%s & $%s$ & $[%s, %s]$ & $%s$ / $%s$ & %s \\\\' %
                         (disp(p) if p in DISPLAY else '\\texttt{%s}' % _tex(p),
                          _sg(d), _sg(dlo), _sg(dhi), _sg(per[BANK1]),
                          _sg(per[BANK2]), 'yes' if rep else '\\textbf{no}'))
            out[(opp, p)] = d
        writeTable('partners_%s.tex' % opp, lines, 'lrcrc',
                   'Partners & SESTINA $-$ v0.6 & 95\\% CI & Per bank & Repl.')
    # S1 as PREREGISTRATION 5.2 draft 3 states it: min, median, self, ratio, and
    # the incumbent's own baseline over v05 on the same eight rows.
    ch = [out[('v05', p)] for p in PORDER if p != 'itself' and ('v05', p) in out]
    self7 = out[('v05', 'itself')]
    med = sorted(ch)[len(ch) // 2] if len(ch) % 2 else (sorted(ch)[len(ch)//2-1] + sorted(ch)[len(ch)//2]) / 2
    emit('vsevenSoneMin', _sg(min(ch)), 'P5-B6.jsonl: minimum FROZEN-INCUMBENT delta over changed-partner rows')
    emit('vsevenSoneMedian', _sg(med), 'P5-B6.jsonl: median changed-partner delta')
    emit('vsevenSoneSelf', _sg(self7), 'P5-B6.jsonl: self-play delta')
    emit('vsevenSoneRatio', _n(med / self7, 3), 'P5-B6.jsonl: median/self ratio for v0.7 over v0.6')
    emit('vsevenSoneRows', str(len(ch)), 'P5-B6.jsonl: changed-partner rows against a v05 opponent')
    emit('vsevenSonePositive', str(sum(1 for x in ch if x > 0)), 'P5-B6.jsonl: changed-partner rows with a positive delta')
    emit('vsevenSoneThreshold', '-1.00', 'PREREGISTRATION 5.2: the S1 collapse threshold')
    emit('vsevenSoneVerdict', 'PASS' if min(ch) >= -1.0 and sum(1 for x in ch if x > 0) >= 5 else 'FAIL',
         'PREREGISTRATION 5.2: S1 verdict, no row below -1.0 and at least five positive')
    # the incumbent as its own control -- v0.6 over v0.5, the same eight settings
    base = {}
    for p in PORDER:
        inc = [r for r in rows if r['opp'] == 'v05' and r['arm'] == 'INCUMBENT' and r['partners'] == p]
        v5 = [r for r in rows if r['opp'] == 'v05' and r['arm'] == 'v05' and r['partners'] == p]
        if len(inc) == 2 and len(v5) == 2:
            base[p] = delta(inc, v5)[0]
    bch = [base[p] for p in base if p != 'itself']
    bmed = sorted(bch)[len(bch)//2] if len(bch) % 2 else (sorted(bch)[len(bch)//2-1] + sorted(bch)[len(bch)//2]) / 2
    emit('vsevenBaseMin', _sg(min(bch)), 'P5-B6.jsonl: v0.6 over v0.5, minimum changed-partner delta')
    emit('vsevenBaseMedian', _sg(bmed), 'P5-B6.jsonl: v0.6 over v0.5, median changed-partner delta')
    emit('vsevenBaseSelf', _sg(base['itself']), 'P5-B6.jsonl: v0.6 over v0.5, self-play delta')
    emit('vsevenBaseRatio', _n(bmed / base['itself'], 3), 'P5-B6.jsonl: v0.6 over v0.5 median/self ratio')
    TABLE_SPEC['regimes.tex'] = ('lrrrr', 'Comparison & Min & Median & Self & Median/self')
    writeTable('regimes.tex', [
        'SESTINA over v0.6 --- holdout & $%s$ & $%s$ & $%s$ & %s \\\\' %
        (_sg(min(ch)), _sg(med), _sg(self7), _n(med / self7, 3)),
        'v0.6 over v0.5 --- holdout & $%s$ & $%s$ & $%s$ & %s \\\\' %
        (_sg(min(bch)), _sg(bmed), _sg(base['itself']), _n(bmed / base['itself'], 3)),
        'SESTINA over v0.6 --- training, stated in advance & $-0.15$ & $+1.26$ & $+2.94$ & 0.428 \\\\',
        'v0.6 over v0.5 --- training, stated in advance & $-0.61$ & $+0.64$ & $+1.35$ & 0.472 \\\\'])
    emit('vsevenOneSeatFive', _sg(out[('v05', 'v06')]),
         'P5-B6.jsonl: a one-seat upgrade among v06 partners, v05 opponent')
    emit('vsevenOneSeatSix', _sg(out[('v06', 'v06')]),
         'P5-B6.jsonl: a one-seat upgrade among v06 partners, v06 opponent')


# ---------------------------------------------------------------- B7, cross-play
def b7_crossplay():
    rows = _rows('P5-B7.jsonl')
    xp = [r for r in rows if r['kind'] == 'crossplay']
    grid, diag, off = {}, [], []
    for i in range(1, 4):
        for j in range(1, 4):
            rs = [r for r in xp if r['arm'] == 'xp%d' % i and r['partners'] == 'xp%d' % j]
            if len(rs) != 2: continue
            m, lo, hi, _ = pool(rs)
            grid[(i, j)] = (m, lo, hi)
            (diag if i == j else off).append(m)
    lines = []
    for i in range(1, 4):
        lines.append('xp%d & %s \\\\' % (i, ' & '.join(
            '$%s$ $[%s, %s]$' % (_sg(grid[(i, j)][0]), _sg(grid[(i, j)][1]), _sg(grid[(i, j)][2]))
            for j in range(1, 4))))
    TABLE_SPEC['crossplay.tex'] = ('lccc', 'Run \\textbackslash\\ partners & xp1 & xp2 & xp3')
    writeTable('crossplay.tex', lines)
    dm, om = sum(diag) / len(diag), sum(off) / len(off)
    emit('vsevenXpDiagonal', _sg(dm), 'P5-B7.jsonl: mean of the three self-partnered cells')
    emit('vsevenXpOffDiagonal', _sg(om), 'P5-B7.jsonl: mean of the six cross-partnered cells')
    # The sign convention has to be stated, because S2 asks whether the
    # OFF-diagonal collapses relative to the diagonal, so the informative
    # direction is off minus diag: negative means cross-play is worse.
    # docs/v07/FINAL-RESULTS.md prints this quantity as "gap -0.01", which is
    # diag minus off; under the convention that answers S2 it is positive.
    # Nothing turns on it -- |gap| is 76x inside the 1.5-point threshold either
    # way -- but the paper states the direction rather than the magnitude alone.
    emit('vsevenXpGap', _sg(om - dm), 'P5-B7.jsonl: OFF-DIAGONAL MINUS DIAGONAL; negative means cross-play is worse')
    emit('vsevenXpGapRecorded', '-0.01',
         'docs/v07/FINAL-RESULTS.md: the same quantity printed with the opposite sign convention')
    emit('vsevenXpGapAbs', _n(abs(om - dm)), 'P5-B7.jsonl: magnitude of the cross-play gap')
    emit('vsevenXpWorstOff', _sg(min(off)), 'P5-B7.jsonl: worst off-diagonal cell')
    emit('vsevenXpBestOff', _sg(max(off)), 'P5-B7.jsonl: best off-diagonal cell')
    emit('vsevenXpWeakestDiag', _sg(min(diag)), 'P5-B7.jsonl: weakest diagonal cell')
    emit('vsevenXpWorstOffShortfall', _n(min(diag) - min(off)),
         'P5-B7.jsonl: weakest diagonal cell minus worst off-diagonal cell')
    emit('vsevenXpHalfWidth', _n((grid[(1, 1)][2] - grid[(1, 1)][1]) / 2), 'P5-B7.jsonl: per-cell half-width')
    emit('vsevenStwoThreshold', '1.50', 'PREREGISTRATION 5.2: the S2 threshold, points')
    emit('vsevenStwoVerdict', 'PASS' if abs(om - dm) < 1.5 else 'FAIL', 'PREREGISTRATION 5.2: S2 verdict')
    h2h = [r for r in rows if r['kind'] == 'h2h']
    hl, sep = [], 0
    PAIRTAG = {(1, 2): 'OneTwo', (1, 3): 'OneThree', (2, 3): 'TwoThree'}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        rs = [r for r in h2h if r['arm'] == 'xp%d' % a and r['opp'] == 'xp%d' % b]
        if len(rs) != 2: continue
        m, lo, hi, _ = pool(rs)
        if lo > 0 or hi < 0: sep += 1
        emit('vsevenHtwoh%s' % PAIRTAG[(a, b)], _sg(m), 'P5-B7.jsonl: xp%d vs xp%d head to head' % (a, b))
        emit('vsevenHtwoh%sLo' % PAIRTAG[(a, b)], _sg(lo), 'P5-B7.jsonl: lower bound')
        emit('vsevenHtwoh%sHi' % PAIRTAG[(a, b)], _sg(hi), 'P5-B7.jsonl: upper bound')
        hl.append('xp%d vs xp%d & $%s$ & $[%s, %s]$ \\\\' % (a, b, _sg(m), _sg(lo), _sg(hi)))
    TABLE_SPEC['crossh2h.tex'] = ('lrr', 'Pair & Edge (pp) & 95\\% CI')
    writeTable('crossh2h.tex', hl)
    emit('vsevenHtwohSeparated', str(sep), 'P5-B7.jsonl: head-to-head pairs whose interval excludes zero')


# ---------------------------------------------------------------- B8, rule dialects
DTAG = {'default': 'Default', 'no-out-of-turn': 'NoOutOfTurn',
        'no-cardless-declare': 'NoCardless', 'maxasks=360': 'Maxasks',
        'arb=high': 'ArbHigh', 'arb=turn': 'ArbTurn', 'sets=8': 'SetsEight', 'legacy': 'Legacy'}

def b8_dialects():
    rows = _rows('P5-B8.jsonl')
    base = pool([r for r in rows if r['row'] == 'default'])[0]
    lines, exc = [], []
    for name in ['default', 'no-out-of-turn', 'no-cardless-declare', 'maxasks=360',
                 'arb=high', 'arb=turn', 'sets=8', 'legacy']:
        rs = [r for r in rows if r['row'] == name]
        if len(rs) != 2: continue
        m, lo, hi, _ = pool(rs)
        per = {r['bank']: _edge(r) for r in rs}
        t = DTAG[name]
        emit('vsevenDia%s' % t, _sg(m), 'P5-B8.jsonl %s: FROZEN vs INCUMBENT in this dialect' % name)
        emit('vsevenDia%sLo' % t, _sg(lo), 'P5-B8.jsonl %s: lower bound' % name)
        emit('vsevenDia%sHi' % t, _sg(hi), 'P5-B8.jsonl %s: upper bound' % name)
        emit('vsevenDia%sVsDefault' % t, _sg(m - base), 'P5-B8.jsonl %s: excursion from the default row' % name)
        if name != 'default': exc.append((abs(m - base), name, m - base))
        lines.append('%s & $%s$ & $[%s, %s]$ & $%s$ / $%s$ & $%s$ \\\\' %
                     (_tex(name), _sg(m), _sg(lo), _sg(hi), _sg(per[BANK3]),
                      _sg(per[BANK1]), _sg(m - base)))
    TABLE_SPEC['dialects.tex'] = ('lrrrr',
        'Rule variant & Edge (pp) & 95\\% CI & Per bank & vs default')
    writeTable('dialects.tex', lines)
    mx = max(exc)
    emit('vsevenDiaMaxExcursion', _sg(mx[2]), 'P5-B8.jsonl: largest excursion from the default row')
    emit('vsevenDiaMaxExcursionRow', _tex(mx[1]), 'P5-B8.jsonl: dialect attaining it')
    emit('vsevenDiaRows', str(len(lines)), 'P5-B8.jsonl: dialect rows measured')
    emit('vsevenDiaRowsWord', _word(len(lines)),
         'P5-B8.jsonl: the same count spelled, for running prose')
    emit('vsevenSfiveTolerance', '2.00', 'PREREGISTRATION 5.2: the S5 tolerance, points')
    emit('vsevenSfiveVerdict', 'PASS' if mx[0] <= 2.0 else 'FAIL', 'PREREGISTRATION 5.2: S5 verdict')
    # the legacy residual: legacy minus the isolable components.  --maxasks=360
    # is bit-identical to default, so the residual is legacy minus TWO effective
    # components and not three, which is why it is printed with that caveat.
    comp = sum(pool([r for r in rows if r['row'] == n])[0] - base
               for n in ('no-out-of-turn', 'no-cardless-declare', 'maxasks=360'))
    leg = pool([r for r in rows if r['row'] == 'legacy'])[0] - base
    # `maxasks=360' is bit-identical to `default' -- equal winRateA and equal ci
    # tuples on both banks -- so its term is identically zero and contributes no
    # variance.  Folding it in leaves legacy + default - n1 - n2, which is the
    # two-component residual the prose asserts, with unit coefficients.
    ses = []
    for n in ('legacy', 'no-out-of-turn', 'no-cardless-declare', 'default'):
        _, _, _, s = pool([r for r in rows if r['row'] == n])
        ses.append(s)
    h = 1.96 * math.sqrt(sum(s * s for s in ses))
    emit('vsevenLegacyResidual', _sg(leg - comp, 3), 'P5-B8.jsonl: legacy minus its isolable components')
    emit('vsevenLegacyResidualLo', _sg(leg - comp - h), 'P5-B8.jsonl: lower bound')
    emit('vsevenLegacyResidualHi', _sg(leg - comp + h), 'P5-B8.jsonl: upper bound')
    emit('vsevenLegacyResidualRatio', _n(2 * h / abs(leg - comp), 1),
         'P5-B8.jsonl: interval width as a multiple of the residual')
    ident = [r for r in rows if r['row'] == 'maxasks=360']
    dflt = [r for r in rows if r['row'] == 'default']
    same = all(_edge(a) == _edge(b) for a in ident for b in dflt if a['bank'] == b['bank'])
    emit('vsevenMaxasksIdentical', 'yes' if same else 'no',
         'P5-B8.jsonl: whether --maxasks=360 is bit-identical to default')


# ---------------------------------------------------------------- B9, negative controls
def b9_controls():
    rows = _rows('P5-B9.jsonl')
    lines = []
    for h in ('0.05', '0.08', '0.11', '0.15'):
        plant = [r for r in rows if r.get('kind') == 'planted-cost' and r.get('hstr') == h]
        hcap = [r for r in rows if r.get('kind') == 'responder-vs-handicapped' and r.get('hstr') == h]
        clean = [r for r in rows if r.get('kind') == 'responder-vs-frozen' and r.get('hstr') == h]
        if len(plant) != 2 or len(hcap) != 2 or len(clean) != 2: continue
        t = {'0.05': 'Hfive', '0.08': 'Height', '0.11': 'Heleven', '0.15': 'Hfifteen'}[h]
        pm, plo, phi, _ = pool(plant)
        hm, hlo, hhi, _ = pool(hcap)
        cm, clo, chi, _ = pool(clean)
        d, dlo, dhi = delta(hcap, clean)
        for nm, (a, b, c) in (('Cost', (pm, plo, phi)), ('Hcap', (hm, hlo, hhi)),
                              ('Clean', (cm, clo, chi)), ('Excess', (d, dlo, dhi))):
            emit('vsevenB9%s%s' % (t, nm), _sg(a), 'P5-B9.jsonl hstr=%s: %s' % (h, nm))
            emit('vsevenB9%s%sLo' % (t, nm), _sg(b), 'P5-B9.jsonl hstr=%s: %s lower bound' % (h, nm))
            emit('vsevenB9%s%sHi' % (t, nm), _sg(c), 'P5-B9.jsonl hstr=%s: %s upper bound' % (h, nm))
        emit('vsevenB9%sRecovered' % t, 'yes' if dlo > 1.53 else 'no',
             'P5-B9.jsonl hstr=%s: is the planted edge recovered above the 1.53 floor' % h)
        lines.append('%s%s & \\ci{$%s$}{%s, %s} & \\ci{$%s$}{%s, %s} & '
                     '\\ci{$%s$}{%s, %s} & \\ci{$\\mathbf{%s}$}{%s, %s} \\\\' %
                     (h, ' \\textit{(sub-floor)}' if h == '0.05' else '',
                      _sg(pm), _sg(plo), _sg(phi), _sg(hm), _sg(hlo), _sg(hhi),
                      _sg(cm), _sg(clo), _sg(chi), _sg(d), _sg(dlo), _sg(dhi)))
    # separate the rungs, but not the last row from \bottomrule
    lines = [l + '\\addlinespace[0.9ex]' for l in lines[:-1]] + lines[-1:]
    writeTable('controls.tex', lines, '@{}l C C C C@{}',
               'Planted handicap & Planted cost & vs.\\ handicapped & vs.\\ SESTINA & Recovered excess',
               xtab=True)
    ident = [r for r in rows if r.get('kind') == 'identity']
    wr = {r['match']['winRateA'] for r in ident}
    ci = {tuple(r['match']['ci']) for r in ident}
    emit('vsevenIdentityRate', '%.6f' % list(wr)[0], 'P5-B9.jsonl B9.4: FROZEN against itself, win rate')
    emit('vsevenIdentityExact', 'yes' if wr == {0.5} and ci == {(0.5, 0.5)} else 'no',
         'P5-B9.jsonl B9.4: exactly 50.000%% with a zero-width interval on both banks')
    emit('vsevenIdentityN', _commafy(ident[0]['match']['games']), 'P5-B9.jsonl B9.4: games per bank')
    # B9.5, the calibrated side channels
    side = _rows('P5-B9side.jsonl')
    # PREREGISTRATION B9.5 names which tests each planted cheat MUST fail, and
    # they are not one each: cheat=seed must fail S4 AND S5.  Printing the
    # required set beside the measured one is what makes the row a control
    # rather than a claim that every cheat trips exactly one test.
    REQUIRED = {'seed': ('S4', 'S5'), 'shared': ('S6',), 'conv': ('S3',)}
    sl = []
    for cheat in ('seed', 'shared', 'conv'):
        rs = [r for r in side if r['cheat'] == cheat]
        res = {}
        for t in ('s3', 's4', 's5', 's6'):
            st = []
            for r in rs:
                sd = r.get('side') or {}
                if t == 's6':
                    # S6 is emitted as a count, not as a named test row, and its
                    # gate condition is the --freshagents pass.  Zero is a pass.
                    if 's6' in sd and isinstance(sd['s6'], dict) and r.get('freshagents'):
                        st.append('PASS' if sd['s6'].get('mismatch', 0) == 0 else 'FAIL')
                    continue
                for k, v in (sd.get('tests') or {}).items():
                    if k.lower().startswith(t):
                        st.append(v.get('status'))
            res[t] = ('FAIL' if 'FAIL' in st else ('PASS' if st else '---'))
            emit('vsevenCheat%s%s' % (cheat.capitalize(), t.upper()), res[t],
                 'P5-B9side.jsonl: cheat=%s, test %s' % (cheat, t.upper()))
        req = REQUIRED[cheat]
        got = tuple(t.upper() for t in ('s3', 's4', 's5', 's6') if res[t] == 'FAIL')
        ok = got == req
        emit('vsevenCheat%sAsRequired' % cheat.capitalize(), 'yes' if ok else 'NO',
             'P5-B9side.jsonl: cheat=%s fails exactly the tests B9.5 requires' % cheat)
        sl.append('\\texttt{v07x:cheat=%s} & %s & %s & %s \\\\' % (
            cheat, ' & '.join(res[t] for t in ('s3', 's4', 's5', 's6')),
            ', '.join(req), '\\checkmark' if ok else '\\textbf{no}'))
    TABLE_SPEC['sidechannel.tex'] = ('lccccl c',
        'Planted channel & S3 & S4 & S5 & S6 & Must fail & As required')
    writeTable('sidechannel.tex', sl)
    emit('vsevenSideCells', str(len(side)), 'P5-B9side.jsonl: side-channel control cells')


# ---------------------------------------------------------------- B10, the S6 residual
def b10_residual():
    rows = _rows('P5-B10.jsonl')
    lines, gate_nonzero, plain_nonzero = [], 0, 0
    for arm in ('FROZEN', 'INCUMBENT', 'F-cheap', 'composite'):
        for bank in (BANK1, BANK2):
            g = [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1-freshagents']
            p = [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1']
            if not g or not p: continue
            gm, gn = g[0]['side']['s6']['mismatch'], g[0]['side']['s6']['nodes']
            pm, pn = p[0]['side']['s6']['mismatch'], p[0]['side']['s6']['nodes']
            gate_nonzero += (gm != 0); plain_nonzero += (pm != 0)
            lines.append('%s & %d & %d / %s & \\textbf{%d} / %s \\\\' %
                         (disp(arm), bank, pm, _commafy(pn), gm, _commafy(gn)))
    TABLE_SPEC['residual.tex'] = ('llrr',
        'Configuration & Bank & \\texttt{-{}-threads=1} & '
        '\\begin{tabular}[b]{@{}r@{}}\\texttt{-{}-threads=1}\\\\\\texttt{-{}-freshagents}\\end{tabular}')
    writeTable('residual.tex', lines)
    emit('vsevenSsixGateNonzero', str(gate_nonzero), 'P5-B10.jsonl: gate-condition cells with a nonzero mismatch')
    emit('vsevenSsixPlainNonzero', str(plain_nonzero), 'P5-B10.jsonl: one-thread cells with a nonzero mismatch')
    emit('vsevenSsixCells', str(len(rows)), 'P5-B10.jsonl: audited cells')
    emit('vsevenSsixDeals', _commafy(sum(r['side']['deals'] for r in rows)),
         'P5-B10.jsonl: deals audited in total')
    searching = {'FROZEN', 'F-cheap', 'composite'}
    nz = {r['arm'] for r in rows if r['cond'] == 'threads1' and r['side']['s6']['mismatch']}
    emit('vsevenSsixAllSearching', 'yes' if nz and nz <= searching else 'no',
         'P5-B10.jsonl: whether every nonzero one-thread count belongs to a searching arm')
    # the denominators move between conditions -- that is --freshagents changing play
    moved = sum(1 for arm in ('FROZEN', 'INCUMBENT', 'F-cheap', 'composite') for bank in (BANK1, BANK2)
                if [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1']
                and [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1-freshagents']
                and [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1'][0]['side']['s6']['nodes']
                != [r for r in rows if r['arm'] == arm and r['bank'] == bank and r['cond'] == 'threads1-freshagents'][0]['side']['s6']['nodes'])
    emit('vsevenSsixDenomMoved', str(moved),
         'P5-B10.jsonl: cells whose audited-decision denominator changes under --freshagents')


# ---------------------------------------------------------------- B0, verification
def b0_verification():
    d = json.load(open(os.path.join(_RES, 'P5-B0.json')))
    emit('vsevenBinarySha', d['binary']['sha256'], 'P5-B0.json: sha256 of the binary that played every cell')
    emit('vsevenBinaryShaShort', d['binary']['sha256'][:16], 'P5-B0.json: first 16 of that hash')
    emit('vsevenBinaryBytes', _commafy(d['binary']['bytes']), 'P5-B0.json: binary size')
    # The recorded commit and its post-rewrite identity are emitted by
    # commit_map(); this name would sanitise to the same macro and collide.
    bd = d['B0_1_bankDigests']
    emit('vsevenBanks', str(len(bd['rows'])), 'P5-B0.json: deal banks whose digest was reproduced')
    emit('vsevenBanksMatch', 'all' if bd['allMatch'] else 'NOT ALL', 'P5-B0.json: bank digests matching commitment')
    emit('vsevenBankDeals', _commafy(bd['rows'][0]['deals']), 'P5-B0.json: deals folded into each digest')
    lines = ['%d & %s & \\texttt{%s} & %s \\\\' % (r['seed'], _commafy(r['deals']), r['digest'],
                                                   '\\checkmark' if r['match'] else '\\textbf{MISMATCH}')
             for r in bd['rows']]
    TABLE_SPEC['banks.tex'] = ('lrlc', 'Bank & Deals & Digest reproduced & Matches commitment')
    writeTable('banks.tex', lines)
    sa = d['B0_2_sealedAdversaries']
    emit('vsevenSealedRows', str(sa['rows']), 'P5-B0.json: rows decoded from the sealed adversary half')
    emit('vsevenSealedSha', sa['sha256'][:16], 'P5-B0.json: sha256 of the decoded plaintext, first 16')
    emit('vsevenSealedMatch', 'yes' if sa['match'] else 'NO', 'P5-B0.json: plaintext matches SEAL.json')
    emit('vsevenSealCommit', sa['sealCommit'][:7], 'P5-B0.json: commit at which the material was sealed')
    fv = d['B0_3_freezeVerify']
    emit('vsevenFreezeVerify', 'PASS' if fv['rc'] == 0 else 'FAIL', 'P5-B0.json: freeze_config_v07.py --verify-only')
    emit('vsevenMirrorDigest', fv['mirrorDigest'], 'P5-B0.json: R3 mirror pathology digest')
    emit('vsevenCoordinates', str(fv['coordinates']), 'P5-B0.json: coordinates in the frozen vector')
    emit('vsevenFrozenSpec', _tex(fv['reconstructedSpec']), 'P5-B0.json: spec reconstructed from the freeze artifact')
    # Individual option values, so a formula in the text can name a search
    # parameter without anyone typing it.  Read from the freeze artifact's own
    # option map, which is the object the spec is reconstructed from.
    try:
        frz = json.load(open(os.path.join(_ROOT, 'engine', 'fishbot_v07.json')))
        OPTNAME = {'r12': 'Contest', 'stall': 'StallK', 'det': 'Det', 'cand': 'Cand',
                   'kappa': 'Kappa', 'depth': 'Depth', 'maxq': 'Maxq', 'rtie': 'Rtie',
                   'force': 'Force', 'rbelief': 'Rbelief'}
        for key, tag in OPTNAME.items():
            if key in frz.get('options', {}):
                emit('vsevenOpt' + tag, _tex(frz['options'][key]),
                     'engine/fishbot_v07.json: frozen option %s' % key)
        # Deployed coordinate values, read out of the frozen vector rather than
        # out of a header default.  factory.hpp maps the flat vector as
        # NFEAT ask weights, then fourteen v0.5 knobs, then three v0.6 ask
        # terms, then eighteen v0.7 responder coordinates; v06.hpp:216-232
        # fixes the offsets inside the knob block.
        ap = frz.get('allparams') or []
        lay = frz.get('allparamsLayout') or {}
        NF = int(lay.get('nfeat', 20))
        if len(ap) >= 55:
            emit('vsevenPriorTheta', '%.5f' % ap[NF + 9],
                 'engine/fishbot_v07.json allparams: deployed priorTheta (v06.hpp:225)')
            emit('vsevenPriorPhi', '%.5f' % ap[NF + 10],
                 'engine/fishbot_v07.json allparams: deployed priorPhi (v06.hpp:226)')
            emit('vsevenLinearWeight', '%.5f' % ap[NF + 6],
                 'engine/fishbot_v07.json allparams: deployed linearWeight (v06.hpp:222)')
            off = NF + int(lay.get('v05knobs', 14)) + int(lay.get('v06askTerms', 3))
            emit('vsevenContestWeight', '%g' % ap[off + 12],
                 'engine/fishbot_v07.json allparams: the contestation coordinate, responder index 12')
            emit('vsevenContestSign', 'positive' if ap[off + 12] > 0 else 'negative',
                 'engine/fishbot_v07.json allparams: its sign')
            nz = [i for i in range(off, len(ap)) if ap[i] != 0]
            emit('vsevenResponderNonzero', str(len(nz)),
                 'engine/fishbot_v07.json allparams: non-zero responder coordinates')
            emit('vsevenResponderTotal', str(int(lay.get('v07responder', 18))),
                 'engine/fishbot_v07.json allparamsLayout: responder coordinates in total')

        # stall2 defaults to twice the stall threshold (v07_stall.hpp)
        if 'stall' in frz.get('options', {}):
            emit('vsevenOptStallTwo', str(2 * int(frz['options']['stall'])),
                 'v07_stall.hpp: stage-2 threshold, twice the stage-1 threshold')
    except Exception as e:
        print('freeze option macros skipped:', e, file=sys.stderr)
    # The spec is one unbreakable control word to TeX, so setting it verbatim
    # overflows the column by ~260pt.  A copy with a discretionary break after
    # every comma typesets; the characters are identical.
    emit('vsevenFrozenSpecWrapped',
         _tex(fv['reconstructedSpec']).replace(',', ',\\allowbreak{}'),
         'P5-B0.json: the same spec, with a discretionary break after each comma')
    r2 = fv.get('R2a_vector_search', {})
    emit('vsevenRtwoaIdentical', 'yes' if r2.get('identical') else 'no',
         'P5-B0.json: R2a round-trip with the search on is identical')
    emit('vsevenRtwoaGames', _commafy(r2.get('games', 0)), 'P5-B0.json: games in the R2a comparison')
    eng = d['B0_5_engine']
    vtxt = eng['verify']['stdout']
    m = _re.search(r'audit violations:\s*(\d+)\s*/\s*(\d+)', vtxt)
    emit('vsevenVerifyViolations', m.group(1), 'P5-B0.json: fish7 verify, audit violations')
    emit('vsevenVerifyChecks', _commafy(m.group(2)), 'P5-B0.json: fish7 verify, checks')
    m = _re.search(r'checks\s+(\d+)', eng['selftest']['stdout'])
    emit('vsevenSelftestChecks', _commafy(m.group(1)), 'P5-B0.json: fish7 selftest, checks')
    emit('vsevenSelftest', 'PASS' if eng['selftest']['pass_'] else 'FAIL', 'P5-B0.json: fish7 selftest verdict')
    # the frozen configuration's option map and vector layout, as tables, so the
    # appendix reproduces the artifact rather than paraphrasing it
    fz = json.load(open(os.path.join(_ROOT, 'engine', 'fishbot_v07.json')))
    KEYDOC = {
        'r12': 'half-suit contestation weight (\\texttt{oppCertDonate})',
        'rtie': 'tie-break by a hash of the public event stream, not sort order',
        'pool': 'urgency pooling threshold; \\texttt{-1} disables',
        'oppfloor': 'urgency opponent-ownership floor; \\texttt{-1} disables',
        'force': 'urgency forcing horizon in events; $10^6$ disables',
        'askfloor': 'urgency ask floor; \\texttt{-1} disables',
        'stall': 'escalate after $K$ public events with no change in this seat\'s certificate hash',
        's1': 'enable the endgame-truncated determinized search',
        'det': 'determinizations sampled per searched decision',
        'cand': 'candidate actions carried into the search',
        'kappa': 'shrinkage of the paired lower-confidence-bound deviation rule',
        'rbelief': 'belief model used to reconstruct continuation players',
        'depth': 'search depth limit in plies',
        'maxq': 'search runs only when at most this many cards remain unresolved',
    }
    opt_lines = []
    for k, v in fz['options'].items():
        opt_lines.append('\\texttt{%s} & \\texttt{%s} & %s \\\\'
                         % (_tex(k), _tex(v), KEYDOC.get(k, '---')))
    writeTable('options.tex', opt_lines, 'llp{0.52\\linewidth}',
               'Key & Value & What it controls')
    # The component table the agent section leads with: mechanism first, harness
    # name second, so a reader meets the idea before the identifier.
    COMPONENTS_TABLE = [
        ('Deal inference', 'Sinkhorn fit to the deal posterior, from a fitted policy prior',
         'unchanged', '\\S\\ref{sec:inference}'),
        ('Ask and declaration policy', 'fitted linear score, 37 coordinates',
         'unchanged; no refit in this cycle', '\\S\\ref{sec:agent-policy}'),
        ('Tie-breaking among equal-scoring asks', 'enumeration order (unstable sort)',
         'hash of the public event stream (\\texttt{rtie})', '\\S\\ref{sec:agent-tiebreak}'),
        ('Half-suit contestation', 'absent',
         'contested-half-suit weight (\\texttt{r12})', '\\S\\ref{sec:agent-contest}'),
        ('Declaration urgency escalation', 'event-count clock at 220 events',
         'disabled (\\texttt{pool}, \\texttt{oppfloor}, \\texttt{force}, \\texttt{askfloor})',
         '\\S\\ref{sec:agent-urgency}'),
        ('Termination safeguard', 'the urgency clock',
         'deduction-state stall detector (\\texttt{stall})', '\\S\\ref{sec:agent-stall}'),
        ('Test-time search', 'built and measured, deployed disabled',
         'enabled, endgame-truncated, paired LCB guard', '\\S\\ref{sec:agent-search}'),
    ]
    writeTable('components.tex',
               ['%s & %s & %s & %s \\\\' % r for r in COMPONENTS_TABLE],
               '@{}Q{0.77} Q{0.92} Q{1.31} l@{}',
               'Component & v0.6 & SESTINA v1.0 & Section', xtab=True)
    lay = fz['allparamsLayout']
    lay_lines = []
    for k, v in lay.items():
        if k == 'note': continue
        # the summary row is a sum of the rows above it, not a sixth block
        if k == 'total': lay_lines.append('\\midrule')
        lay_lines.append('%s & %s \\\\' % (('%s' if k == 'total' else '\\texttt{%s}') % _tex(k), _tex(v)))
    writeTable('layout.tex', lay_lines, 'lr', 'Block & Coordinates')
    # the note names a source file; every other filename in the paper is
    # typewriter, and a roman \_ prints as a long low bar
    note = _re.sub(r'((?:[A-Za-z0-9./-]|\\_)+\.(?:hpp|cpp|py|json|tex))',
                   r'\\texttt{\1}', _tex(lay.get('note', '')))
    emit('vsevenLayoutNote', note, 'engine/fishbot_v07.json: allparamsLayout note')
    emit('vsevenNotVectorisable', ', '.join('\\texttt{%s}' % _tex(x)
                                            for x in fz['switchesNotExpressibleAsVector']),
         'engine/fishbot_v07.json: switches that cannot be expressed in the parameter vector')
    emit('vsevenInheritedVector', _tex(fz['inheritedVectorProvenance']),
         'engine/fishbot_v07.json: provenance of the inherited parameter vector')
    rep = d['reproducibility']
    emit('vsevenReproRuns', str(len(rep['runs'])), 'P5-B0.json: repeat runs of one training cell')
    emit('vsevenReproIdentical', 'yes' if rep['identical'] else 'no',
         'P5-B0.json: whether all repeat runs were bit-identical at 400 deals')
    emit('vsevenSeedViolations', str(len(d['B0_4_seeds']['registryViolations'])),
         'P5-B0.json: pre-existing seed-registry violations reported')
    # The source-drift figure has to come from the artifact.  An earlier draft
    # of this paper transcribed "five" out of prose that the phase-5 recording
    # has since superseded; the artifact says nine, and which nine matters.
    dp = os.path.join(_RES, 'P5-drift.json')
    if os.path.exists(dp):
        dr = json.load(open(dp))
        frz = [k for k in dr if k.startswith('sourcesDifferingFromFreezeCommit')]
        base = dr.get('sourcesDifferingFromArtifactBaseline', [])
        if frz:
            diff = dr[frz[0]]
            emit('vsevenDriftChanged', str(len(diff)), 'P5-drift.json: sources differing from the freeze commit')
            emit('vsevenDriftTotal', str(dr['nSources']), 'P5-drift.json: engine sources compared')
            emit('vsevenDriftList', ', '.join('\\texttt{%s}' % _tex(x) for x in diff),
                 'P5-drift.json: which sources differ from the freeze commit')
            emit('vsevenDriftMidBattery', str(len(base)),
                 'P5-drift.json: sources differing from the artifact baseline, i.e. arriving mid-battery')
            emit('vsevenDriftPreBattery', str(len(diff) - len(base)),
                 'P5-drift.json: sources that already differed before the battery began')


# ---------------------------------------------------------------- B1, the commit gate
GTAG = ['G1 dead asks', 'G2 longest dead run', 'G3 games w/ run>=6', 'G4 action-limit',
        'G5 mirror tail', 'G6 late declarations', 'G7a S3/S4/S5', 'G7b S6 seat-isolation']
GNAME = {'G1 dead asks': 'provably-dead asks $\\le 0.10\\%$',
         'G2 longest dead run': 'longest dead run $\\le 5$',
         'G3 games w/ run>=6': 'games with a run $\\ge 6$ = 0',
         'G4 action-limit': 'action-limit games = 0',
         'G5 mirror tail': 'mirror tail max $< 220$, p99 $\\le 150$',
         'G6 late declarations': 'declarations at/after event 220 = 0',
         'G7a S3/S4/S5': 'S3/S4/S5 certified, zero tolerance',
         'G7b S6 seat-isolation': 'S6 = 0 at \\texttt{-{}-threads=1 -{}-freshagents}'}
CTAG = {'FROZEN': 'Frozen', 'INCUMBENT': 'Incumbent', 'F-cheap': 'Fcheap', 'NEGCONTROL': 'Neg'}
RTAG = {'G1': 'Gone', 'G2': 'Gtwo', 'G3': 'Gthree', 'G4': 'Gfour', 'G5': 'Gfive',
        'G6': 'Gsix', 'G7a': 'Gsevena', 'G7b': 'Gsevenb'}

def b1_gate():
    rows = load_jsonl(os.path.join(_RES, 'P5-gate.jsonl'))
    by = {r['id']: r for r in rows}
    order = ['FROZEN', 'INCUMBENT', 'F-cheap', 'NEGCONTROL']
    lines = []
    for g in GTAG:
        cells = []
        for c in order:
            ok = by[c]['rules'].get(g)
            cells.append('ok' if ok else '\\textbf{fail}')
            emit('vsevenGate%s%s' % (CTAG[c], RTAG[g.split()[0]]), 'ok' if ok else 'fail',
                 'P5-gate.jsonl: %s, rule %s' % (c, g.split()[0]))
        lines.append('%s & %s \\\\' % (GNAME[g], ' & '.join(cells)))
    lines.append('\\midrule\n\\textbf{verdict} & %s \\\\' %
                 ' & '.join('\\textbf{%s}' % by[c]['verdict'] for c in order))
    TABLE_SPEC['gate.tex'] = ('lcccc', 'Rule & SESTINA v1.0 & v0.6 & \\texttt{F-cheap} & Negative control')
    writeTable('gate.tex', lines)
    for c in order:
        emit('vsevenGate%sVerdict' % CTAG[c], by[c]['verdict'], 'P5-gate.jsonl: %s gate verdict' % c)
    fz = by['FROZEN']
    emit('vsevenGateRules', str(len(GTAG)), 'P5-gate.jsonl: rules in the commit gate')
    emit('vsevenGateRulesWord', _word(len(GTAG)),
         'P5-gate.jsonl: the same count spelled, for running prose')
    emit('vsevenGateGames', _commafy(fz['games']), 'P5-gate.jsonl: mirror games per configuration')
    emit('vsevenGateDeadAsks', '%.5f' % fz['stats']['deadAskPct'], 'P5-gate.jsonl FROZEN: provably-dead asks, percent')
    emit('vsevenGateLongestDead', str(fz['stats']['longestDead']), 'P5-gate.jsonl FROZEN: longest dead run')
    emit('vsevenGateEvents', _n(fz['stats']['eventsMean'], 3), 'P5-gate.jsonl FROZEN: events per game')
    emit('vsevenGateTailMax', str(fz['stats']['eventsMax']), 'P5-gate.jsonl FROZEN: mirror tail maximum')
    emit('vsevenGateTailPnn', str(fz['stats']['eventsP99']), 'P5-gate.jsonl FROZEN: mirror tail p99')
    emit('vsevenGateAskHit', _n(fz['stats']['askHit'], 3), 'P5-gate.jsonl FROZEN: ask hit rate, percent')
    emit('vsevenGateMisdecl', _n(fz['stats']['declPct'], 3), 'P5-gate.jsonl FROZEN: misdeclaration, percent')
    emit('vsevenGateSsix', str(fz['s6']['mismatch']), 'P5-gate.jsonl FROZEN: S6 mismatches at one thread, fresh agents')
    emit('vsevenGateSsixNodes', _commafy(fz['s6']['nodes']), 'P5-gate.jsonl FROZEN: S6 audited decisions')
    neg = by['NEGCONTROL']
    failed = [g.split()[0] for g in GTAG if not neg['rules'].get(g)]
    emit('vsevenNegFails', ', '.join(failed), 'P5-gate.jsonl: rules the negative control fails')
    emit('vsevenNegFailCount', str(len(failed)), 'P5-gate.jsonl: how many rules the negative control fails')
    emit('vsevenNegDeadAsks', _n(neg['stats']['deadAskPct'], 5), 'P5-gate.jsonl NEGCONTROL: provably-dead asks, percent')
    emit('vsevenNegLongestDead', str(neg['stats']['longestDead']), 'P5-gate.jsonl NEGCONTROL: longest dead run')
    emit('vsevenNegTailMax', str(neg['stats']['eventsMax']), 'P5-gate.jsonl NEGCONTROL: mirror tail maximum')


# ---------------------------------------------------------------- B11, selection bias
def b11_selection():
    # K configurations were scored against v06 before the freeze; at the lattice
    # cell size the expected maximum of K draws under the null that none differs
    # is sigma*sqrt(2 ln K).  PREREGISTRATION B11 fixes K and the cell size.
    K, N = 15, 24000
    sigma = 98.0 / 2.0 / math.sqrt(N)
    emit('vsevenSelK', str(K), 'PREREGISTRATION B11: configurations scored against v06 before the freeze')
    emit('vsevenSelSigma', _n(sigma, 4), 'PREREGISTRATION B11: per-cell sigma at the lattice cell size')
    emit('vsevenSelExpectedMax', _n(sigma * math.sqrt(2 * math.log(K)), 2),
         'PREREGISTRATION B11: sigma * sqrt(2 ln K), the expected maximum under the null')
    emit('vsevenSelOnHoldout', '0.00',
         'PREREGISTRATION B11: selection term on holdout, zero by construction -- the banks were never available to select on')
    # the third clause: is the holdout estimate short of the training one, and where.
    b2 = _rows('P5-B2.jsonl')
    grp = pool([r for r in b2 if r['cell'] == 'B2.4'])[0]
    b6 = _rows('P5-B6.jsonl')
    fr = [r for r in b6 if r['opp'] == 'v05' and r['arm'] == 'FROZEN' and r['partners'] == 'itself']
    inc = [r for r in b6 if r['opp'] == 'v05' and r['arm'] == 'INCUMBENT' and r['partners'] == 'itself']
    b6self = delta(fr, inc)[0]
    b7 = _rows('P5-B7.jsonl')
    xp = [r for r in b7 if r['kind'] == 'crossplay']
    dg = [pool([r for r in xp if r['arm'] == 'xp%d' % i and r['partners'] == 'xp%d' % i])[0] for i in range(1, 4)]
    og = [pool([r for r in xp if r['arm'] == 'xp%d' % i and r['partners'] == 'xp%d' % j])[0]
          for i in range(1, 4) for j in range(1, 4) if i != j]
    # the training column is TRANSCRIBED from the protocol, which states these in
    # advance for exactly this comparison.  Its source is named so the provenance
    # check can see it.
    SRC = 'docs/v07/PREREGISTRATION.md'
    train = [('B2.4 --- \\texttt{rtie} + urgency-off + \\texttt{stall}, as a group', 0.78, grp, 'Group'),
             ('B6 self-play row, opponent \\texttt{v05}', 2.94, b6self, 'Bsix'),
             ('B7 diagonal (self-play)', 4.51, sum(dg) / 3, 'BsevenDiag'),
             ('B7 off-diagonal (cross-play)', 4.48, sum(og) / 6, 'BsevenOff')]
    lines, short = [], []
    for label, tr, ho, tag in train:
        emit('vsevenSelTrain%s' % tag, _sg(tr), '%s: training figure stated in advance' % SRC)
        emit('vsevenSelHold%s' % tag, _sg(ho), 'P5-B2/B6/B7: the holdout measurement of the same quantity')
        emit('vsevenSelShort%s' % tag, _sg(ho - tr), 'holdout minus training')
        if ho - tr < 0: short.append((ho - tr, label, tag))
        lines.append('%s & $%s$ & $%s$ & $%s$ \\\\' % (label, _sg(tr), _sg(ho), _sg(ho - tr)))
    TABLE_SPEC['selection.tex'] = ('lrrr', 'Quantity & Training & Holdout & Shortfall')
    writeTable('selection.tex', lines)
    emit('vsevenSelShortCount', str(len(short)), 'how many of the four quantities fall short of their training value')
    emit('vsevenSelReproduce', str(len(train) - len(short)), 'how many reproduce or exceed their training value')
    if short:
        s = min(short)
        emit('vsevenSelWorstShortfall', _n(-s[0]), 'the largest shortfall, in points')
        emit('vsevenSelWorstTag', s[2], 'which quantity attains it')


# ---------------------------------------------------------------- throughput
def throughput_paper():
    """gamesPerSec is WHOLE-MATCH throughput and carries the opponent's cost on
    both sides of the ratio, so INCUMBENT_gps / ARM_gps is strictly below
    t_arm/t_inc.  Every figure emitted here is therefore a LOWER BOUND, and the
    tightest bound is against the cheapest opponent."""
    rows = _rows('P5-B3.jsonl')
    gps = {}
    for r in rows:
        gps.setdefault((r['arm'], r['opp']), []).append(r['match']['gamesPerSec'])
    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    cls = {r['opp']: r['cls'] for r in rows}
    lines = []
    for arm, tag in (('FROZEN', 'Frozen'), ('F-cheap', 'Fcheap'), ('composite', 'Composite')):
        ratios = {}
        for (a, p), v in gps.items():
            if a != arm: continue
            inc = gps.get(('INCUMBENT', p))
            if inc: ratios[p] = med(inc) / med(v)
        tight = ratios.get('random')
        far = med([r for p, r in ratios.items() if cls[p] == 'far'])
        allm = med(list(ratios.values()))
        emit('vsevenCost%sTight' % tag, _n(tight), 'P5-B3.jsonl: gamesPerSec ratio against random, the tightest lower bound')
        emit('vsevenCost%sFar' % tag, _n(far), 'P5-B3.jsonl: median ratio over the far archetypes')
        emit('vsevenCost%sPanel' % tag, _n(allm), 'P5-B3.jsonl: median ratio over the whole panel, the most diluted')
        lines.append('%s & %s$\\times$ & %s$\\times$ & %s$\\times$ \\\\' % (disp(arm), _n(tight), _n(far), _n(allm)))
    # This table aggregates as a ratio of medians (per panel member, the
    # incumbent's median throughput over the arm's, then the median of those).
    # engine/p5_analyse.py medians the per-bank ratios instead.  Neither is
    # canonical and they differ; a report that records divergences from the
    # phase-5 recording as corrections has to measure this one rather than
    # eyeball it, so the largest divergence across the six median cells is
    # computed and printed.
    worst = 0.0
    for arm in ('FROZEN', 'F-cheap', 'composite'):
        for sel in (lambda p: cls[p] == 'far', lambda p: True):
            rom, per_bank = [], []
            for (a, p), v in gps.items():
                if a != arm or not sel(p) or ('INCUMBENT', p) not in gps:
                    continue
                inc = gps[('INCUMBENT', p)]
                rom.append(med(inc) / med(v))
                # pair by bank, not by rank: sorting the two lists independently
                # can divide one bank's incumbent by the other bank's arm
                per_bank += [i / j for i, j in zip(inc, v)]
            if rom:
                worst = max(worst, abs(med(rom) - med(per_bank)))
    emit('vsevenCostAggDelta', _n(worst, 3),
         'P5-B3.jsonl: largest divergence between this table\'s aggregation and a median of per-bank ratios')
    TABLE_SPEC['cost.tex'] = ('lrrr',
        'Configuration & vs \\texttt{random} (tightest) & Far archetypes (median) & Whole panel (median)')
    writeTable('cost.tex', lines)
    emit('vsevenCostRecorded', '3.2', 'docs/v07/PREREGISTRATION.md: the cost multiple the record carried')


# Identifiers in the deviation register.  The source document backticks some of
# them and not others, so a pass over the un-backticked remainder is what makes
# the column consistent: --threads=13, main.cpp and a bare digest are all set the
# way the rest of the paper sets an identifier.  Segments already inside a
# \texttt group are left alone rather than nested.
_TTSPLIT = _re.compile(r'(\\texttt\{[^}]*\})')

def _tt_bare(seg):
    seg = _re.sub(r'(?<![-\w])--([A-Za-z][A-Za-z0-9-]*(?:=[^\s,;.)]+)?)',
                  r'\\texttt{-{}-\1}', seg)
    seg = _re.sub(r'(?<![\w/])((?:[A-Za-z0-9./-]|\\_)+\.(?:hpp|cpp|py|json|jsonl|md|txt|tex)'
                  r'(?::\d+)?)(?![\w])', r'\\texttt{\1}', seg)
    seg = _re.sub(r'(?<![\w])([0-9a-f]{16,})(?![\w])', r'\\texttt{\1}', seg)
    return seg

def _tt_identifiers(body):
    return ''.join(p if p.startswith('\\texttt{') else _tt_bare(p)
                   for p in _TTSPLIT.split(body))


# ---------------------------------------------------------------- deviations
def deviations():
    """The deviation register is READ OUT of docs/v07/FINAL-RESULTS.md rather
    than retyped, so the paper's appendix cannot drift from the results
    document it reports."""
    p = os.path.join(_ROOT, 'docs', 'v07', 'FINAL-RESULTS.md')
    txt = open(p).read()
    # the kind may itself name a deviation ("CORRECTION TO D8"), so digits are
    # allowed inside it; without that, D18 is silently dropped from the register.
    # The kind may carry emphasis markup and a cross-reference of its own
    # ("OBSERVATION, **SUPERSEDED BY D19**"), so the character class has to admit
    # asterisks and commas.  Without that D7 is silently dropped, the register
    # loses an entry another entry points at, and the printed count is the
    # regex's yield rather than the document's.
    ds = _re.findall(r'\*\*(D\d+)\*\*\s+([A-Z][A-Z*, \-0-9]*?)\s+--\s+(.*?)(?=\n\n\*\*D\d+\*\*|\n\n---)',
                     txt, _re.S)
    found = {d[0] for d in ds}
    declared = set(_re.findall(r'^\*\*(D\d+)\*\*', txt, _re.M))
    if declared - found:
        raise SystemExit('deviation register: %s present in the source but not captured'
                         % ', '.join(sorted(declared - found)))
    lines = []
    kinds = {}
    for did, kind, body in ds:
        kind = kind.replace('*', '').strip().rstrip(',')
        kinds[kind] = kinds.get(kind, 0) + 1
        body = ' '.join(body.split())
        body = body.replace('\\', '').replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
        body = body.replace('#', '\\#').replace('$', '\\$')
        # Elide before converting the Markdown markup below, not after: cutting
        # inside a \\texttt{...} group would emit an unbalanced brace.
        if len(body) > 460:
            # Elide at a sentence boundary so that every printed entry ends with
            # a complete sentence; fall back to a word boundary only if no
            # sentence ends after the first 200 characters.
            cut = body[:460]
            k = cut.rfind('. ')
            body = cut[:k + 1] if k >= 200 else cut[:457].rsplit(' ', 1)[0]
            # an elision can strand the opening half of a pair; drop it rather
            # than let it swallow the rest of the entry
            for ch in ('"', '`'):
                if body.count(ch) % 2:
                    body = body[:body.rfind(ch)].rstrip()
            body += '\\,\\ldots'
        # Code spans FIRST, quotes second.  The other order eats itself: `` is
        # what a converted opening quote is made of, so a code-span pass run
        # afterwards matches it as an empty span and emits \\texttt{}.
        # Markdown code spans would otherwise typeset as two opening quotes,
        # `like this`, because LaTeX has no closing backtick; set them the way
        # the paper sets every other identifier.
        body = _re.sub(r'`([^`]*)`', r'\\texttt{\1}', body)
        # The register is read out of a Markdown document, which uses straight
        # quotes.  Every other quotation in the paper is typographic, so balanced
        # pairs are converted rather than left to render as two verticals.
        body = _re.sub(r'"([^"]*)"', lambda m: '``' + m.group(1) + "''", body)
        # Options, file names and digests the source did not backtick.
        body = _tt_identifiers(body)
        # A long option keeps two hyphens.  `--' ligatures to an en dash, so
        # --threads=13 was reaching the page as -threads=13, which is a different
        # command.  A `--' with space on both sides is prose punctuation and is
        # left alone.
        body = _re.sub(r'(?<![-\w])--(?=[A-Za-z])', '-{}-', body)
        if body.count('{') != body.count('}'):
            raise SystemExit('deviation %s: unbalanced braces after conversion: %r' % (did, body))
        for junk in ('\\texttt{}', '``\'\'', '\"'):
            if junk in body:
                raise SystemExit('deviation %s: %r survived conversion in %r' % (did, junk, body))
        lines.append('\\textbf{%s} & %s & %s \\\\' % (did, _tex(kind).title(), body))
    writeTable('deviations.tex', lines, '@{}l l L@{}',
               'ID & Kind & Deviation, addition or correction', longtable=True, xtab=True)
    emit('vsevenDeviations', str(len(ds)), 'docs/v07/FINAL-RESULTS.md section 15: recorded deviations')
    emit('vsevenDeviationCorrections', str(sum(v for k, v in kinds.items() if k.startswith('CORRECTION'))),
         'docs/v07/FINAL-RESULTS.md section 15: entries that are corrections')
    emit('vsevenDeviationAdded', str(sum(v for k, v in kinds.items() if k.startswith('ADDED'))),
         'docs/v07/FINAL-RESULTS.md section 15: entries that add a cell, a check or an artifact')


# ---------------------------------------------------------------- battery totals
def totals():
    files = ['P5-B2.jsonl', 'P5-B3.jsonl', 'P5-B4eval.jsonl', 'P5-B5.jsonl', 'P5-B6.jsonl',
             'P5-B7.jsonl', 'P5-B8.jsonl', 'P5-B9.jsonl']
    cells, games, viol, limit = 0, 0, 0, 0
    for fn in files:
        for r in _rows(fn):
            cells += 1
            games += r['match']['games']
            viol += r['match'].get('auditViolations', 0)
            limit += 1 if r['match'].get('limitHitRate', 0) else 0
    emit('vsevenScoredCells', str(cells), 'P5-B2..B9: scored match cells')
    emit('vsevenScoredGames', _commafy(games), 'P5-B2..B9: games played in scored cells')
    emit('vsevenAuditViolations', str(viol), 'P5-B2..B9: audit violations across the whole battery')
    # `match` does not run the soundness audit, so auditChecks is 0 in every
    # scored cell and "zero violations" over the battery is zero out of zero.
    # The engine-wide audit figure comes from B0.5 and is cited there alone.
    emit('vsevenAuditChecks', str(sum(r['match'].get('auditChecks', 0)
                                      for fn in files for r in _rows(fn))),
         'P5-B2..B9: audit CHECKS across the battery -- zero, i.e. no audit evidence')
    emit('vsevenLimitCells', str(limit), 'P5-B2..B9: cells with any action-limit hit')
    lim = [r for fn in files for r in _rows(fn) if r['match'].get('limitHitRate', 0)]
    if lim:
        emit('vsevenLimitCellGames', _commafy(lim[0]['match']['games']),
             'P5-B3.jsonl: games in the one cell carrying an action-limit hit')
        emit('vsevenLimitCellArm', _tex(lim[0].get('arm', '?')), 'P5-B3.jsonl: its arm')
        emit('vsevenLimitCellOpp', _tex(lim[0].get('opp', '?')), 'P5-B3.jsonl: its opponent')
    # the cell counts per battery, against the preregistered design
    # The PREREGISTERED column is the count the protocol names, NOT the count
    # that ran.  B5 and B9 differ: deviation D1 adds a twelfth B5 cell (the
    # FROZEN reference the leave-one-out drops are taken from, without which no
    # drop is computable) and D2 adds sixteen B9 cells (the planted cost and the
    # responder-vs-unhandicapped cell, without which "tracks the planted size"
    # and "resolves to zero" are not computable).  Printing the as-run count in
    # both columns would make every battery look complete by construction.
    design = [('B2', 'headline strength against the frontier', 'P5-B2.jsonl', 10, ''),
              ('B3', 'the shared 31-member panel', 'P5-B3.jsonl', 248, ''),
              ('B4', 'a fresh adversary search', 'P5-B4eval.jsonl', 16, ''),
              ('B5', 'the attribution lattice', 'P5-B5.jsonl', 22, 'D1'),
              ('B6', 'the partner-regime table', 'P5-B6.jsonl', 64, ''),
              ('B7', 'cross-play between independent runs', 'P5-B7.jsonl', 24, ''),
              ('B8', 'the rule-dialect table', 'P5-B8.jsonl', 16, ''),
              ('B9', 'the negative controls', 'P5-B9.jsonl', 10, 'D2'),
              ('B10', 'the S6 residual', 'P5-B10.jsonl', 16, '')]
    lines, dropped, added = [], 0, 0
    for bid, what, fn, exp, dev in design:
        got = len(_rows(fn))
        dropped += max(0, exp - got)
        added += max(0, got - exp)
        lines.append('%s & %s & %d & %d & %s & %s \\\\' % (
            bid, what, exp, got - exp if got > exp else 0, got,
            ('complete' if got >= exp else '\\textbf{INCOMPLETE}') + (' (%s)' % dev if dev else '')))
    emit('vsevenCellsAdded', str(added), 'cells added beyond the preregistered design, per D1 and D2')
    writeTable('battery.tex', lines, '@{}l L r r r l@{}',
               'Battery & What it measures & Registered & Added & Measured & Status', xtab=True)
    emit('vsevenCellsDropped', str(dropped), 'preregistered cells not measured')
    # CORRECTION.  docs/v07/FINAL-RESULTS.md states "409 scored match cells" in
    # three places; the figure is hand-typed at engine/p5_analyse.py:1135 and is
    # not computed from the artifacts.  Counting rows that carry a `match`
    # object across B2, B3, B4eval, B5, B6, B7, B8 and B9 gives the number
    # emitted above.  No verdict depends on it -- it is a denominator for D22 --
    # but the paper prints the counted figure and says so.
    emit('vsevenScoredCellsRecorded', '409',
         'docs/v07/FINAL-RESULTS.md: the hand-typed cell count this generator corrects')


def commit_map():
    """Commit identity after the history rewrite.

    An oversized capture artifact was excised from git history so the repository
    could be pushed, which rewrote every commit from phase 3 onward.  The hashes
    recorded inside the phase-5 artifacts therefore name objects that no longer
    resolve.  Rather than reprint a dead hash or retype a live one, each recorded
    commit is re-found here by its subject line and the current hash emitted
    beside the recorded one, with the lookup asserted unique.
    """
    import subprocess

    def git(*a):
        return subprocess.run(['git'] + list(a), cwd=_ROOT, capture_output=True,
                              text=True).stdout.strip()

    def alive(h):
        return git('cat-file', '-t', h) == 'commit'

    def by_subject(pat):
        out = git('log', '--all', '--format=%h\x1f%s')
        hits = [l.split('\x1f') for l in out.split('\n') if l and pat.lower() in l.split('\x1f')[1].lower()]
        return hits[0][0] if len(hits) == 1 else ''

    emit('vsevenHeadCommit', git('rev-parse', '--short', 'HEAD'),
         'git rev-parse HEAD: the commit this build was made at')

    # the commit the B0 verification block recorded
    try:
        rec = json.load(open(os.path.join(_RES, 'P5-B0.json')))['commit'][:7]
    except Exception:
        rec = ''
    if rec:
        emit('vsevenBzeroCommitRecorded', rec,
             'P5-B0.json: the commit recorded at generation time, pre-rewrite')
        cur = rec if alive(rec) else by_subject('cross-play re-run on the actual frozen architecture')
        if cur:
            emit('vsevenBzeroCommit', cur,
                 'git log: the same commit after the history rewrite, found by subject line')
        emit('vsevenBzeroCommitAlive', 'yes' if alive(rec) else 'no',
             'whether the recorded hash still resolves in the rewritten history')

    # the freeze commit, which the drift artifact keys on
    frz_rec = ''
    try:
        d = json.load(open(os.path.join(_RES, 'P5-drift.json')))
        for k in d:
            m = _re.search(r'([0-9a-f]{7,40})$', k)
            if m:
                frz_rec = m.group(1)[:7]
                break
    except Exception:
        pass
    if frz_rec:
        emit('vsevenFreezeCommitRecorded', frz_rec,
             'P5-drift.json: the freeze commit as recorded, pre-rewrite')
        cur = frz_rec if alive(frz_rec) else by_subject('FREEZE -- one configuration')
        if cur:
            emit('vsevenFreezeCommit', cur,
                 'git log: the freeze commit after the history rewrite, found by subject line')

    # the seal commit predates the rewrite and is unchanged; assert that
    try:
        seal = json.load(open(os.path.join(_ROOT, 'research', 'v07', 'banks',
                                           'SEAL.json')))['commit'][:7]
        emit('vsevenSealCommitAlive', 'yes' if alive(seal) else 'no',
             'whether the seal commit survives the history rewrite')
    except Exception:
        pass

    # the excised artifact, cited through its digest sidecar
    side = os.path.join(_RES, 'K5-capture-7030004.sha256')
    if os.path.exists(side):
        txt = open(side).read()
        m = _re.search(r'sha256:\s*([0-9a-f]{64})', txt)
        if m:
            emit('vsevenKfiveDigest', m.group(1)[:16],
                 'K5-capture-7030004.sha256: digest of the excised capture, first 16')
        m = _re.search(r'size:\s*(\d+)', txt)
        if m:
            emit('vsevenKfiveBytes', _commafy(m.group(1)),
                 'K5-capture-7030004.sha256: size of the excised capture')


def manifest_table():
    man = json.load(open(os.path.join(_RES, 'MANIFEST-P5.json')))
    emit('vsevenManifestRuns', str(len(man['runs'])), 'MANIFEST-P5.json: digested artifacts')
    ok = 0
    for r in man['runs']:
        fp = os.path.join(_ROOT, r['artifact']) if os.path.exists(os.path.join(_ROOT, r['artifact'])) \
             else os.path.join(_RES, os.path.basename(r['artifact']))
        if os.path.exists(fp) and hashlib.sha256(open(fp, 'rb').read()).hexdigest() == r['sha256']:
            ok += 1
    emit('vsevenManifestVerified', str(ok), 'MANIFEST-P5.json: artifacts whose sha256 re-verifies on disk')
    emit('vsevenManifestFrozenSpec', _tex(man['frozen_spec']), 'MANIFEST-P5.json: the frozen spec')


# ---------------------------------------------------------------- the verdict table
def verdicts():
    """PREREGISTRATION section 6's seven conditions and section 5's claims, each
    with the artifact that settles it.  Emitted as a table so the paper cannot
    quietly restate a threshold."""
    b2 = _rows('P5-B2.jsonl')
    fc = [r for r in b2 if r['cell'] == 'B2.2']
    m, lo, hi, _ = pool(fc)
    certified = lo > 1.53 and replicates(fc)
    # The protocol's own label for the top outcome was "certified", which reads
    # as third-party verification. The paper names what was actually satisfied.
    emit('vsevenPrimaryVerdict',
         ('satisfies the registered primary criterion' if certified
          else ('a measured but unconfirmed advance' if lo > 0 and replicates(fc)
                else 'no advance')),
         'PREREGISTRATION 5.1 applied to P5-B2.jsonl B2.2')
    emit('vsevenBtwoFcheapWinRate', '%.2f\\%%' % (50.0 + m),
         'P5-B2.jsonl B2.2: pooled head-to-head win rate against F-cheap, 50 + edge')
    emit('vsevenPrimaryFloor', '1.53', 'PREREGISTRATION 5.1: the phase-2 C1-class detection floor')
    emit('vsevenDeclFloor', '2.13', 'PREREGISTRATION 3 correction 4: the declaration-family floor')
    comp = [r for r in b2 if r['cell'] == 'B2.4']
    cm, clo, chi, _ = pool(comp)
    emit('vsevenItemTwoMet', 'MET' if clo <= 0 <= chi else 'not met',
         'PREREGISTRATION 6 item 2 applied to P5-B2.jsonl B2.4')


def forest_plot():
    """Figure 3: the principal comparisons as a forest plot, drawn from the
    artifacts so the figure cannot drift from the tables.  TikZ coordinates are
    computed here rather than typed into the .tex.

    Two panels.  The upper panel is the five direct head-to-head comparisons,
    each SESTINA's edge against the named configuration.  The lower panel holds
    two different estimands and is labelled as such: the worst partner
    substitution, which is a difference between two upgrade effects, and
    SESTINA's edge against the strongest adversarial response, which is the
    NEGATION of the adversary's own edge as Table 22 prints it.  Both panels are
    oriented so that positive favours SESTINA; the raw B4 data and Table 22's
    convention are unchanged."""
    direct = []
    b2 = _rows('P5-B2.jsonl')
    for cellid, label in (('B2.1', 'vs.\\ v0.6 (deployed)'),
                          ('B2.2', 'vs.\\ F-cheap \\textit{(primary)}'),
                          ('B2.3', 'vs.\\ F-mid'),
                          ('B2.4', 'vs.\\ phase-2 composite'),
                          ('B2.5', 'vs.\\ v0.5')):
        rs = [r for r in b2 if r['cell'] == cellid]
        m, lo, hi, _ = pool(rs)
        direct.append((label, m, lo, hi, cellid == 'B2.2'))
    other = []
    b6 = _rows('P5-B6.jsonl')
    fr = [r for r in b6 if r['opp'] == 'v05' and r['arm'] == 'FROZEN' and r['partners'] == 'withholder']
    inc = [r for r in b6 if r['opp'] == 'v05' and r['arm'] == 'INCUMBENT' and r['partners'] == 'withholder']
    d, dlo, dhi = delta(fr, inc)
    other.append(('worst partner substitution', d, dlo, dhi, False))
    b4 = _rows('P5-B4eval.jsonl')
    worst = max(((pool([r for r in b4 if r['cell'] == z])[2], z) for z in {r['cell'] for r in b4}))
    rs = [r for r in b4 if r['cell'] == worst[1]]
    m, lo, hi, _ = pool(rs)
    # Table 22 prints the ADVERSARY's edge; here SESTINA's edge against it, so
    # the value and both bounds are negated and the bounds swap.
    other.append(('vs.\\ strongest adversary', -m, -hi, -lo, False))

    rows = direct + other
    lo_ax = min(r[2] for r in rows) - 0.6
    hi_ax = max(r[3] for r in rows) + 0.6
    span = hi_ax - lo_ax
    W = 8.0   # narrower than the single-panel figure: the row values now carry their intervals
    x = lambda v: (v - lo_ax) / span * W
    out = ['\\begin{tikzpicture}[x=1cm,y=1cm]']
    step_y = 0.72
    ys = []
    y = 0.0
    for i, (label, m, lo, hi, primary) in enumerate(direct):
        ys.append(y); y -= step_y
    # Panel geometry, top to bottom: last A row, a clear band, the separator,
    # a clear band, the B title (set above its anchor), a clear band, the first
    # B row.  Each band is wide enough that no text touches the rule.
    y_sep = ys[-1] - 0.55
    y_titleB = y_sep - 0.50          # anchor (south west) of the B title
    y = y_titleB - 0.42
    for i, (label, m, lo, hi, primary) in enumerate(other):
        ys.append(y); y -= step_y
    out.append('  \\node[anchor=south west,font=\\scriptsize\\itshape,text=fishgray] at (0,%.3f) {A. direct comparisons: SESTINA\'s edge against the named configuration};' % (ys[0] + 0.30))
    out.append('  \\node[anchor=south west,font=\\scriptsize\\itshape,text=fishgray] at (0,%.3f) {B. other estimands, oriented so that positive favours SESTINA};' % (y_titleB))
    for (label, m, lo, hi, primary), yy in zip(rows, ys):
        col = 'fishgreen' if primary else 'fishgray'
        w = 'ultra thick' if primary else 'thick'
        out.append('  \\node[anchor=east,font=\\small] at (-0.25,%.3f) {%s};' % (yy, label))
        out.append('  \\draw[%s,%s] (%.3f,%.3f) -- (%.3f,%.3f);' % (col, w, x(lo), yy, x(hi), yy))
        out.append('  \\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);' % (col, x(lo), yy - 0.11, x(lo), yy + 0.11))
        out.append('  \\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);' % (col, x(hi), yy - 0.11, x(hi), yy + 0.11))
        out.append('  \\fill[%s] (%.3f,%.3f) circle (0.075);' % (col, x(m), yy))
        out.append('  \\node[anchor=west,font=\\scriptsize\\ttfamily] at (%.3f,%.3f) {%+.2f [%+.2f, %+.2f]};'
                   % (W + 0.25, yy, m, lo, hi))
    ybot = ys[-1] - 0.55
    ytop = ys[0] + 0.62
    out.append('  \\draw[fishgray,thin] (-4.2,%.3f) -- (%.3f,%.3f);' % (y_sep, W + 3.4, y_sep))
    out.append('  \\draw[densely dashed,fishgray] (%.3f,%.3f) -- (%.3f,%.3f);' % (x(0), ybot, x(0), ytop))
    out.append('  \\draw[->,fishgray] (0,%.3f) -- (%.3f,%.3f);' % (ybot, W + 0.1, ybot))
    step = 2 if span > 9 else 1
    t = int(math.ceil(lo_ax / step) * step)
    while t <= hi_ax:
        out.append('  \\draw[fishgray] (%.3f,%.3f) -- (%.3f,%.3f) node[below,font=\\scriptsize] {%d};'
                   % (x(t), ybot, x(t), ybot - 0.1, t))
        t += step
    out.append('  \\node[font=\\scriptsize,anchor=north] at (%.3f,%.3f) {win-rate edge (percentage points); positive favours SESTINA};'
               % (W / 2, ybot - 0.45))
    out.append('\\end{tikzpicture}')
    with open(os.path.join(_TBL, 'forest.tex'), 'w') as f:
        f.write('\n'.join(out) + '\n')


# ---------------------------------------------------------------- ask features and weights
# Appendix B promises the twenty ask features with their fitted weights.  The
# names are the source's own (engine/src/v05.hpp, the w[NFEAT] comment column)
# and the weights are the first NFEAT coordinates of the frozen allparams
# vector, read from the freeze artifact rather than typed.
ASK_FEATURES = ['hit probability', 'squared hit', 'certain hit', 'own set progress',
                'team control', 'lock completion', 'continuation', 'completion bonus',
                'reply threat', 'information leak', 'target hand size', 'empties target',
                'repeats set', 'known team cards', 'location entropy', 'team owns set',
                'exposure on miss', 'trailing pressure', 'runway', 'leak magnitude']

def ask_weights():
    fz = json.load(open(os.path.join(_ROOT, 'engine', 'fishbot_v07.json')))
    w = fz['allparams']
    layout = fz.get('allparamsLayout', {})
    nfeat = int(layout.get('nfeat', 20)) if isinstance(layout, dict) else 20
    if nfeat != len(ASK_FEATURES):
        raise SystemExit('ask_weights: layout says %d ask weights, names list has %d' % (nfeat, len(ASK_FEATURES)))
    lines = []
    for i, name in enumerate(ASK_FEATURES):
        lines.append('%d & %s & $%s$ \\\\' % (i, _tex(name), _sg(w[i], 5)))
    writeTable('askweights.tex', lines, 'rlr', 'Index & Feature & Weight')
    emit('vsevenAskWeightCount', str(nfeat), 'engine/fishbot_v07.json: ask-weight coordinates')


# ---------------------------------------------------------------- K2 declaration counts
# The phase-3 replay of the declaration allocators recorded every voluntary
# declaration of the captured team.  Two counts are emitted: every recorded
# declaration across the three capture files, and the count over distinct
# (bank, urgency-setting) cells, which excludes the two re-runs on the same
# seeds and the two cells of the sequential-joint policy.  The paper quotes both.
def k2_declaration_counts():
    files = {'on': 'K2-ceiling-urgon.jsonl', 'off': 'K2-ceiling-urgoff.jsonl', 'jalloc': 'K2-perdec-jalloc.jsonl'}
    total = 0
    best = {}
    for tag, fn in files.items():
        for r in load_jsonl(os.path.join(_RES, fn)):
            n = int(r['metrics']['declShipAccuracy']['n'])
            total += n
            if tag in ('on', 'off'):
                key = (tag, r['seed'])
                if key not in best or r['deals'] > best[key][0]:
                    best[key] = (r['deals'], n)
    distinct = sum(n for _, n in best.values())
    emit('vsevenLoneDeclsTotal', _commafy(total),
         'K2-ceiling-urgon/urgoff, K2-perdec-jalloc: recorded voluntary declarations, all capture cells')
    emit('vsevenLoneDeclsDistinct', _commafy(distinct),
         'K2-ceiling-urgon/urgoff: declarations over distinct (bank, urgency) cells, largest cell per pair')


def paper_main():
    os.makedirs(_TBL, exist_ok=True)
    b3_panel_table()          # the headline: worst case and minimax regret, first
    b2_frontier()
    b4_adversaries()
    b5_attribution()
    b6_partners()
    b7_crossplay()
    b8_dialects()
    b9_controls()
    b10_residual()
    b11_selection()
    b0_verification()
    b1_gate()
    throughput_paper()
    deviations()
    totals()
    manifest_table()
    commit_map()
    verdicts()
    forest_plot()
    ask_weights()
    k2_declaration_counts()
    emit('vsevenBuildDate', _time.strftime('%Y-%m-%d'), 'build_tables_v07.py --paper run date')
    # provenance counts, which the reproducibility appendix quotes
    try:
        secs = os.path.join(_PAPER, 'sections_v07')
        used7, usedold = set(), set()
        for fn in os.listdir(secs):
            if not fn.endswith('.tex'): continue
            tx = open(os.path.join(secs, fn)).read()
            used7 |= set(_re.findall(r'\\(vseven[A-Za-z]+)', tx))
            usedold |= set(_re.findall(r'\\(vsix[A-Za-z]+|num[A-Za-z]+)', tx))
        gen = set(_re.findall(r'renewcommand\{\\(vseven[A-Za-z]+)\}', '\n'.join(OUT)))
        # emitted below, so they belong to `gen` for the completeness check
        gen |= {'vsevenProvGenerated', 'vsevenProvInherited', 'vsevenProvTranscribed'}
        emit('vsevenProvGenerated', str(len(used7 & gen)),
             'macros used in sections_v07 that this script generates from artifacts')
        # The earlier cycles' generators emit \vsix and \num macros from their
        # own artifacts. Those are generated, not transcribed, and counting them
        # as transcribed understates the discipline while overstating the hand
        # work; counting them as generated-by-this-cycle would overstate this
        # cycle. They get their own number.
        inh = set()
        for fn in ('numbers_v05_generated.tex', 'numbers_v06_generated.tex'):
            fp = os.path.join(_PAPER, fn)
            if os.path.exists(fp):
                inh |= set(_re.findall(r'renewcommand\{\\((?:vsix|num)[A-Za-z]+)\}', open(fp).read()))
        emit('vsevenProvInherited', str(len(usedold & inh)),
             'macros used in sections_v07 generated from artifacts by an EARLIER cycle')
        # A v0.7 transcribed number carries the \vseven prefix like every other
        # macro in this manuscript, so counting by prefix returns zero and
        # asserts that nothing is transcribed.  The count is used-minus-generated,
        # which is what paper/check_provenance.py reports.
        emit('vsevenProvTranscribed', str(len((used7 | usedold) - gen - inh)),
             'macros used in sections_v07 that no generator emits, i.e. transcribed by hand')
        # A macro that numbers_v07.tex declares under a source header is
        # TRANSCRIBED from an earlier phase on purpose, so it is not missing.
        # Only a \vseven macro that nothing declares is a typesetting bug.
        declared = set()
        ph = os.path.join(_PAPER, 'numbers_v07.tex')
        if os.path.exists(ph):
            declared = set(_re.findall(r'providecommand\{\\(vseven[A-Za-z]+)\}', open(ph).read()))
        missing = sorted(used7 - gen - declared)
        if missing:
            print('WARNING: %d \\vseven macro(s) used in sections_v07 are NOT generated: %s'
                  % (len(missing), ', '.join(missing[:20])), file=sys.stderr)
    except FileNotFoundError:
        pass
    path = os.path.join(_PAPER, 'numbers_v07_generated.tex')
    with open(path, 'w') as f:
        f.write('% GENERATED by engine/build_tables_v07.py --paper -- do not edit.\n')
        f.write('% Every macro names the artifact and field it came from; paper/check_provenance.py\n'
                '% reads those comments.  No number in sections_v07/ is typed by hand.\n\n')
        f.write('\n'.join(OUT) + '\n')
    print('wrote %d macros to %s' % (len(OUT), path))
    print('wrote %d tables to %s' % (len(os.listdir(_TBL)), _TBL))


if __name__ == '__main__':
    main()
