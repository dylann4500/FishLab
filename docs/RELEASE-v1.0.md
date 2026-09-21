# SESTINA v1.0

> **Revision history (21 September 2026).** The technical report attached to this release was
> revised after two editorial reviews; the revised build (`output/pdf/sestina_v10.pdf`, tag
> `sestina-v1.0-paper-r1`) carries an unversioned title, a complete rule statement, corrected
> claim scopes, and an explicit statement that its confidence intervals are nominal because the
> scored implementation retains agent state across deals. The text below is the record of the
> original v1.0 release and describes the original 72-page build; see
> `docs/SESTINA-v1.0-EDITORIAL-REVISION.md` and `docs/SESTINA-v1.0-ASTRA-DISPOSITION.md` for the
> change logs. The frozen agent, its configuration and every measurement are unchanged.

The first release under the SESTINA name, and the strongest agent this project has produced for
six-player Canadian Fish (Literature). It was developed and evaluated as **FishBot v0.7**, the
seventh cycle of the lineage, and renamed at release; its predecessors keep the names they were 
published under.

**The technical report is attached below as `sestina_v10.pdf` (72 pages).**

## What it is

A frozen configuration, recorded with its 55-coordinate parameter vector in
`engine/fishbot_v07.json`:

```
v07:r12=25,rtie=1,pool=-1,oppfloor=-1,force=1000000,askfloor=-1,stall=12,
    s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26
```

It combines an approximate Sinkhorn fit to the deal posterior started from a fitted policy prior, a
linear ask and declaration policy, a public-history tie-break that preserves common knowledge among
teammates, a half-suit contestation weighting, a deduction-state stall detector in place of an
event-count termination rule, and a guarded determinized test-time search.

## What was measured

On sealed holdout material, under a protocol registered before any of it was played, with
deal-clustered bootstrap intervals and replication across two disjoint deal banks required in
advance.

| comparison | edge | 95% CI | games |
|---|---:|---|---:|
| **vs `F-cheap`** — the registered target | **+3.33 pp** | [+2.88, +3.78] | 48,000 |
| vs the deployed v0.6 policy | +4.63 pp | [+4.19, +5.06] | 48,000 |
| vs `F-mid` | +2.89 pp | [+2.00, +3.78] | 12,000 |
| vs v0.5 | +5.18 pp | [+4.56, +5.81] | 24,000 |
| **vs the phase-2 composite** | **+0.15 pp** | **[−0.29, +0.59]** | 48,000 |

The registered rule required three things at once — a pooled lower bound above the 1.53 pp detection
floor, a sign replicating on both banks, and a pass of the soundness gate. All three were satisfied;
the banks read +3.67 and +2.99.

`F-cheap` was named as the bar rather than the deployed v0.6 policy on purpose: the deployed policy
ships its search off, so beating it is the easier claim.

## What it does not claim

**SESTINA v1.0 does not measurably outperform a composite configuration assembled earlier in the same
programme** (+0.15 pp, interval containing zero), so the architecture work that followed added no
measurable strength. That was one of seven conditions of non-confirmation named in the protocol
before any holdout was played, and it is recorded as having been met.

Also reported, at full strength:

- Over a shared 31-member opponent panel the worst cell is −0.04 pp [−1.41, +1.33], which does not
  replicate in sign, and it is 3rd of four on minimax regret.
- The attribution location test does not replicate — one bank locates the gain in a single component,
  the other does not.
- Under partner substitution 6 of 7 changed-partner rows stay positive; the worst is −0.19, which
  that battery does not resolve.
- Four candidate mechanisms failed to produce measurable improvement at this resolution.

The report keeps three senses of "strongest" apart and claims only **lineage strength**. Robustness
is evaluated extensively with mixed results. Global standing — whether this is the strongest Canadian
Fish agent anyone has built — is neither established nor claimed. It is not shown to be near-optimal
and not shown to be unexploitable; every exploitability figure here is a lower bound produced by a
search, not a bound on what a search could find.

## What is in this release

| asset | what it is |
|---|---|
| `sestina_v10.pdf` | the technical report, 72 pages |
| `sestina_v10_standalone.tex` | single-file LaTeX source for Overleaf |
| `fishbot_v07.json` | the freeze artifact: the exact configuration, its vector, and its round-trip record |

## Reproducing it

```bash
cd engine && make                                   # clang++ -std=c++20 -O3, produces ./fish
./fish verify --games=600
./fish match --a='v07:r12=25,rtie=1,pool=-1,oppfloor=-1,force=1000000,askfloor=-1,stall=12,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26' \
             --b=v06 --games=400 --rotations=6 --seed=90210 --json
```

Binaries are not committed. A fresh `make` reproduces the play of the binary the published
evaluation ran on: across six cells against v0.6 and v0.5 at three seeds, all 34 reported fields
agree exactly.

Every number in the report is a macro rather than a typed digit — 274 generated directly from
artifacts, 146 transcribed under a comment header naming the source document — and
`paper/check_provenance.py --version v07` fails the build if any number lacks an attributed artifact
that exists on disk. Every recorded result row carries the literal `argv` it was produced by, so any
single cell can be re-run by hand.

## A note on the identifiers

Every seed, deal bank, digest, directory and configuration string still carries a `v07` identifier,
including the freeze artifact and the spec string above. Those names are load-bearing: the banks were
committed under them, the freeze artifact round-trips through them, and the digests are digests of
objects named that way. Renaming them after the fact would break the provenance chain the results
rest on, so they are left exactly as they were sealed. Throughout, *SESTINA v1.0* names the agent and
*the v0.7 cycle* names the programme that produced and measured it.

The name changed at release; nothing measured did.
