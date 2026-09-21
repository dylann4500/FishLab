# SESTINA v1.0 manuscript: editorial revision record (September 2026)

**Input:** `SESTINA_v1_editorial_audit.md` (review dated 21 September 2026) against
`sestina_v10 (1).pdf` (73 pages, dated 31 August 2026).
**Branch:** `paper/sestina-v10-editorial-revision`, from `main` at `f9b4743`.
**Scope:** the audit's first-pass revision (heading map, run-in labels, targeted edits,
replacement passages, T01–T20 claim-precision items, Figure 3). The optional second-pass
condensation (audit §7.2) was not attempted.

This document separates (1) wording changes, (2) scientific corrections with the evidence
that decided them, (3) structural and formatting changes, (4) changes to the table
generator, (5) the checks actually executed, (6) unresolved author queries, and (7) what
was deliberately not done.

## 0. Version check

The reviewed PDF corresponds to the repository source at `f9b4743` plus one uncommitted
working-tree change (a `\clearpage` before the contents in `paper/sestina_v10.tex`). A
baseline build of that source with `tectonic 0.17.0` reproduced 73 pages, and every
search anchor in the audit was found in the modular source, so there are no substantive
version differences to report. The revised build is 82 pages; the growth is in §7
(three interval constructions and the calibration section are now described in full),
§2 (the rule statement is complete), Appendix B (feature-weight table) and Appendix C
(three-column criteria table).

## 1. Wording changes (editorial)

- **Title.** `SESTINA v1.0: Constraint-Based Inference and Empirical Evaluation in
  Six-Player Canadian Fish` (the versioned form, since the series, file names and running
  head carry the version). `pdftitle` updated to match.
- **Complete numbered-heading map** applied as specified (audit §2); no section
  renumbered, no label changed. RQ prefixes removed from headings; the RQ-to-section
  mapping at the start of §11 is retained.
- **Run-in labels** (audit §3) replaced throughout, including §2.3, §4.3, §5.4, §6.3–6.8,
  §7.3, §8, §9.2–9.3, §11.1, §11.3, §11.8, §12.6–12.7, Appendices A, B, C, G, and the
  E.1/E.2 entry labels in the order the audit gives. The archived preregistration
  quotation in Appendix C is untouched.
- **Replacement passages** 5.2–5.13 implemented with the existing numerical macros in
  place of every displayed number (abstract; §1 opening; version nomenclature;
  contributions; §4.2; §4.4; §3 opening; §9.4 opening; §11.1 opening; §11.2.1; §12.3;
  §13).
- **Targeted edits E01–E90** implemented, with these notes: E07 became "the rule set
  adopted for this implementation" (no external rule source is recorded); E56 now states
  the agreement count and its scope; E76 renamed `Zeight` to `Z08` in prose (the
  `\vsevenZeight…` macro names are artifact identifiers and are unchanged); E89 states the
  count was corrected from 409 to 428 via the existing macros.
- **Manuscript-wide style rules** (audit §7.4): "worth zero", "nothing", "closed",
  "exhausted", "locally flat", "genuinely", "honest", "escape hatch", "hack", "excuse",
  "plumbing", "declined" removed from prose (the word "honest" survives only inside the
  quoted protocol provision); interpretive bold removed; "dialect" replaced by "rule
  variant" except in labels, file names and the `--legacy` switch text; `pp` and the
  bracketed 95% CI convention defined once in §7.1; "blueprint", "certificate", "cell",
  "bank" and "frontier" defined at first use.

## 2. Scientific corrections (T01–T20)

Each VERIFY item was checked against the engine source, the generator, the phase
documents and the artifacts before the text was changed. Classification: **confirmed**
(claim supported as written), **qualified** (supported only with a narrower scope, now
stated), **unsupported** (no support found; text narrowed and an author query raised).

| Item | Classification | Evidence (file:line or artifact) | Change made |
|---|---|---|---|
| T01 exact constraint vs. posterior | qualified | `engine/src/blockdp.hpp:19–26,157–176,279–319` counts constraint-consistent deals with unit weight and no likelihood term; the behavioural weighting enters only as the Sinkhorn initialiser (`belief.hpp:526`) and as an importance weight on exact samples in the search (`v06.hpp:694–702`); `v07_invert.hpp:197–276` computes a one-step exact likelihood then accumulates per-card evidence additively | Eq. (1) now defines $q_s^t(D)$ with $\mathcal C_s^t$; text states it is not the policy-aware posterior; "not available in the same object" replaced by the implementation fact; "policy prior" renamed behavioural weighting (identifier kept); §6.8 and §11.8.1 say the inverter approximates (24) |
| T02 decision procedure | qualified | `v05.hpp:804–891` (marginal-product selection), `belief.hpp:580–598` (sequential pricing), `v05.hpp:620–639` (live-ask filter and fallback), `v05.hpp:1045,1067` (team-ownership floor), `v06.hpp:802` ($\kappa_r=0$ in ties), `v06.hpp:683,705` (zero-draw fallback to the v0.5 path); the "$1-5\times10^{-4}$" forced-endgame clause corresponds to `v05.hpp:1123`, a voluntary-path branch unreachable in the frozen configuration | §6.9 steps 2, 3, 5 rewritten; team-ownership floor and forced-endgame ladder added to §6.4; the unreachable threshold clause removed (query Q1) |
| T03 marginal product vs. joint | qualified | `belief.hpp:600–604` states the two are not equivalent in general; `docs/v07/CANDIDATES.md:270–272` is the only "algebraic" statement and supplies no argument; `K2-records-v06-7030001.csv` gives 218/454 over 17,978 declarations on one cell; the $-1.58$ pp figure is inferred via the ledger's 1.2 (1.0–1.4) conversion (`SUBOPTIMALITY-LEDGER.md:57–63`, `k2_flat.py:66`); `\vsevenLoneDecls` counts records over cells that re-run the same seeds | "Algebraic" justification deleted everywhere; empirical agreement retained with scope; denominator, units and inferred conversion stated via new transcribed macros; three allocation procedures named and distinguished; non-uniqueness of the exact argmax and the lexicographic tie noted |
| T04 interval descriptions | qualified; sign error **confirmed** | `arena.hpp:292–309` (percentile bootstrap, B=20,000, deal blocks); `build_tables_v07.py:470–492` (`pool`, `delta`); no per-deal outcomes are stored in any P5 artifact; the stored `ci`/`wilsonCI` pairs show the clustered interval is typically no wider than the independent one | §7.3 rewritten as three labelled constructions; sign claim removed; "conservative" conditioned on non-negative covariance; the legacy-residual qualification now appears in §11.7 and §12.6 |
| T05 CI width vs. power | confirmed as the audit describes | `v07_power.hpp:36–39` is $98/\sqrt N$; no power calculation exists anywhere; the 17–39× ratio compares planning sample sizes on different channels under assumed conversion factors (`CANDIDATES.md:428–435`, `K4_deff_analyse.py:70–90`) | §7.4 renamed and rewritten as a planning approximation; §9.4.3, §12.7 and Appendix G reworded accordingly |
| T06 calibration threshold | qualified | 1.53 pp is the planted cost of the smallest recovered `leak` rung of the phase-2 ladder (`ADVERSARIES.md:144–176`), against a v0.6 target, C1 responder class, 12×16×250 budget, 48,000 games/bank vs 12,000; the preregistration itself calls it "a reference bar, not a detection floor" for direct comparisons (`PREREGISTRATION.md:195–204`); phase-5 rungs recovered on lower-bound reading: 0.11 and 0.15 only | §7.9 rewritten to separate the three quantities and give the floor's origin and scope; §9.2, §10.4, §11.9, §12.3 and E.1 reworded; "responder is empty" replaced by class- and budget-specific statements |
| T07 Figure 3 | fixed | generator `forest_plot()` | Two panels: five direct comparisons; then the partner upgrade-difference and SESTINA's edge against the strongest adversary (negated from Table 22's convention, stated in the caption). Raw B4 data and Table 22 unchanged |
| T08 "strongest" | fixed | §11.2.1, §11.3 | Replaced by "the evaluated configuration" / "a configuration developed from v0.6"; frontier defined as the set {v0.6 deployed, F-cheap, F-mid} ordered by cost (no dominance criterion was applied); F-cheap defined from `PREREGISTRATION.md:159` as v0.6 with the search enabled at SESTINA's search settings |
| T09 public policy, randomness | qualified; implementation completed in the second pass (A3: partial public-event digest) | tie-break `v06.hpp:560,639,656–657`: $\rho$ hashes only the public stream, seat and event count; $\varepsilon=10^{-9}$ absolute (`v06.hpp:47`); the index is applied to the sort-ordered tied block; reset seed derived from bank seed, deal index and seat (`game.hpp:264`, `arena.hpp:176`), private to the seat; persistent state lives in the rollout engine (`RESEARCH-LOG.md §4.3`) | §2.3 and §8 describe the cited correlation model precisely; §6.2 reports $\varepsilon$ and the sort-arrangement dependence; §6.8 restricts (24) to the idealised non-search policy and states the seed and persistence exceptions; §6.9 no longer claims all decisions depend only on the listed information |
| T10 isolation tests | qualified; second pass extends the consequence to interval coverage (A1: every interval nominal) | all scored cells ran at 13 threads with reused agents (`MANIFEST-P5.json`, D17); S6 on holdout was run at one thread only (`P5-B10.jsonl`); no control cell bounds the residual's effect on a win rate | "no side-channel finding" replaced by the audit's suggested wording; Table 24 caption states neither column is the scored mode; new transcribed macro `\vsevenScoredThreads{13}` (source: `FINAL-RESULTS.md`); §7.8/§10.3 describe the gate as an empirical behavioural test suite; corrected-implementation reruns listed as future work |
| T11 exploitability | fixed | — | "adversarial response win-rate edge" used throughout; §8 defines utility, parity benchmark under the rotation, information access (each adversary seat sees its own hand), homogeneity as identical specifications separate from channel prohibitions; "lower bound" language replaced by observed-edge language; Z08 described as an inverter using a v0.6 surrogate |
| T12 verdict interpretation | confirmed | S1 rule is point-estimate based (`PREREGISTRATION.md:527`, `build_tables_v07.py:950–951`); S7 is not computed anywhere and the `PASS` is literal text; condition 2 is interval-based, 4 and 7 point-estimate based | Table 11 gains a "Reading" column; Appendix C gains registered-rule / recorded / reading-with-uncertainty columns, restores the dropped phrases "about 1.5 cell half-widths" and "at or above 1.53 points", and states which verdicts are generator-computed |
| T13 notation, DP, numerics | qualified | Eq. (11) is a product of independent marginals (`v06.hpp:382–389`); exact and deployed allocation probabilities are different routines (`blockdp.hpp:486–511` vs `belief.hpp:580–598`); deployed team probability is $\max(\prod_c\sum_{p\in T}\hat\mu,\hat\pi^\star)$ (`v05.hpp:1036,1066`); candidate sets shrink for unresolved cards and a transferred card leaves $U_t$ (`belief.hpp:213–220`); block tables are `float`, convolutions `double`, no rescaling (`blockdp.hpp:45–76`); self-test tolerances $10^{-9}$ (disjunction-free) and $0.05$ (rejection sampling) (`main.cpp:464`); 0.41 ms is a mean over 6,715 contested-ask builds in 150 self-play games (`E8-ties.txt`, `probe_v06.hpp:125–132`) on the M5 Pro | Notation table gains $q_s^t$, $\mathcal C_s^t$, $\hat\pi_{\rm alloc}$, $\hat\pi_{\rm team}$; Eq. (11) called an independence proxy; explicit sum for $\pi_{\rm team}$ (new Eq. `eq:team`); recurrence boundary conventions and induction argument; floating-point evaluation and tolerances stated; timing scope stated; "inference as solved" replaced |
| T14 search guard | qualified | `v06.hpp:685–708,788–805`: at most 12 accepted worlds, `ess ≤ 1.5 ⇒ se = 1e9`, strictly positive LCB required, index 0 is sort-first, candidates extended through ties up to 20, zero surviving worlds fall back to the v0.5 scoring path; the record measures deficit removal (`E12-search.jsonl`) and a holdout add-in of +1.91 with a leave-one-out straddling zero | §6.7 describes the rule as an LCB-style heuristic without coverage guarantees, states the degenerate-case behaviour, the reference action, and that the record shows deficit reduction, not guaranteed improvement |
| T15 oracle comparisons | unsupported as a game-value bound (both oracles); measurements confirmed; urgency figures condition-specific (second pass A4, B3) | the transfer "oracle" is a greedy argmax of immediately certain pulls on v0.4 mirror self-play, diverging from the unilateral rule in 29 of 6,000 games (`probe_turnxfer_game.hpp:146–206`, `P8-coordination.md`); the arbitration oracle selects among proposals already made, on three of five seeds (`probe_vdeclarb_game.hpp:197–208`, `P6-verify-arbitration-cost.md:64–87`) | "cannot beat the clairvoyant one" removed; both described as oracle-assisted diagnostics with their objectives, policies and seed counts; units kept separate; §9.4.5 now says the incidence-raising route is untested rather than closed |
| T16 attribution language | fixed | — | "most of the gain is individual" → "consistent with benefits beyond identical-agent coordination"; cross-play contrast labelled descriptive (no interval); contestation mechanism "consistent with"; largest drop labelled descriptive; failed location criterion kept adjacent to the attribution summary; "locally flat" removed |
| T17 panel extrema | confirmed | `build_tables_v07.py:596–598,645–656`; a zero self-cell can only lower the worst cell and raise regret; regrets unaffected (none attained at a self-cell); 31 named members are 28 distinct policies (`P5-B3.jsonl`) | §7.11 corrected and now defines $R(a)$ over the four configurations and fixed panel; extrema and rankings labelled descriptive with the CI caveat; 31/28 stated in §7.11, §11.3 and Appendix D |
| T18 rule completeness | confirmed in source; text was incomplete; one appendix row was wrong | `game.hpp:520–529,587` (turn transfer), `game.hpp:424–448` (lowest seat index across both teams; polled between events), `fish.hpp:126–127` and `game.hpp:457–480` (ladder 0.995…0.50 then compulsory), `game.hpp:491–507` (3–3 to the lowest-card holder's team), `game.hpp:597` and `arena.hpp:217` (4–4 credited to team 0; no tie count), `fish.hpp:213` (eight cards per player at `--sets=8`), `main.cpp:61–62` (`--arb=high` is highest seat first) | §2.1–2.2 now state these rules; Appendix A rows corrected (`--arb=high`, deck row, ladder, cap, legacy bundle contents); "completely" removed |
| T19 literature and bibliography | done as editorial; second pass adds search dates, indexes and queries, locators and access dates (B2, B6) | — | §3 opening replaced; loaded words removed; novelty claims scoped to the project's searches; Ryser/Glynn distinguished from approximation algorithms in §3.4 and §5.4; A-loss-recall and degeneracy claims narrowed to what is established; annotations removed from entries 24, 27, 33, 36, 45; entry 63 no longer bundles the ACPC rules; entry 70 split into `nelo` and `nelo-fishtest`; entry 1's date range now ends in 2026; the "four author lists" count corrected to three (per the bibliography's own header) |
| T20 reproducibility | qualified; second pass corrects the repository URL, cites a revision tag, and removes the unavailability-by-construction claim (A7, B10) | features `v05.hpp:430–453`, weights = first 20 coordinates of `allparams` in `engine/fishbot_v07.json`; value-function weights are compiled constants (`v05.hpp:100`); seal check `v07_seeds.hpp:184–226`, bypasses per D5/D6; full binary SHA-256 and mirror digest present, commit hashes short; tag `sestina-v1.0` exists and was uncited; D24 follows D27 with no explanation; Table 26 duplicated Table 7; environment recorded only in `RESEARCH-LOG.md:17`, `MANIFEST.json`, `Makefile` | Appendix B gains the generated feature-weight table and the value-function description; §7.6 and Figure 2 state the seal's scope; Appendix G separates the three identities, cites the tag, adds a computing-environment paragraph, replaces the duplicate table with a cross-reference; Appendix F notes the D24 ordering; cost figures called observed whole-match slowdowns |

## 3. Structural and formatting changes

- Contents restricted to sections and subsections (`tocdepth=2`); sub-subsections remain
  in the PDF bookmarks.
- Figure 1 (pipeline) labels the deployed inference as approximate constraint-based
  inference with behavioural weighting and defines "blueprint".
- Figure 2 moves the training-bank identifiers into the caption.
- Table 3 header "What it is" → "Definition"; Table 15 uses conceptual component names
  with harness identifiers in parentheses; Table 11 and the Appendix C table gain the
  reading columns described above; captions of Tables 8, 12, 13, 15, 16, 20, 22, 23, 24
  and 25 now state metric, direction, units, sample unit, aggregation and interval method.
- Appendix E.3's cross-reference for the sign rule now points to §7.5.
- New transcribed macros in `paper/numbers_v07.tex` (each under a header naming its
  source, as the provenance checker requires): `\vsevenDeclToWinConv{,Lo,Hi}`,
  `\vsevenJointCellDecls`, `\vsevenScoredThreads`, `\vsevenMachineCores`.

## 4. Generator changes (`engine/build_tables_v07.py`)

- `forest_plot()`: two-panel layout with explicit direction; row values now print their
  intervals; axis narrowed to fit the text block.
- `COMPONENTS`: conceptual names with identifiers for the attribution table.
- `vsevenInheritedVector`: full provenance string (previously cut at 120 characters,
  mid-token).
- New `vsevenBtwoFcheapWinRate` (50 + pooled B2.2 edge) and `vsevenAskWeightCount`.
- New `ask_weights()` emitting `paper/tables_v07/askweights.tex` from
  `engine/fishbot_v07.json` and the feature names in `engine/src/v05.hpp`.
- No measurement is changed: the regenerated `numbers_v07_generated.tex` differs from the
  committed one only in the items above, the build date, the transcribed-number tally,
  and `\vsevenHeadCommit` (which tracks the repository HEAD).

## 5. Checks executed

| Check | Result |
|---|---|
| Baseline build, `tectonic --keep-logs paper/sestina_v10.tex` (0.17.0) | exit 0, 73 pages |
| Revised build, same command | exit 0, 82 pages, 0 overfull boxes, 0 undefined references or citations, 3 underfull hboxes (cosmetic, in tables) |
| Flattened source, `python3 paper/inline.py sestina_v10.tex sestina_v10_standalone.tex`, then built | exit 0, 82 pages |
| `python3 engine/build_tables_v07.py --paper` | 620 macros, 27 tables; diff reviewed (see §4) |
| `python3 paper/check_provenance.py --version v07` | passes: every transcribed number attributed to an artifact on disk |
| Label cross-check (every `\ref` target defined in sections and tables) | no unresolved labels |
| Visual inspection of rendered pages 1, 6, 24, 29, 30, 44, 47, 54, 59, 68, 70, 72, 80 | no clipped content; Figure 3 separator overlap found on the first render and fixed |
| `paper/check.py` | not run: it has no v0.7 profile |

No experiment was rerun, no engine source was changed, and no artifact was modified.
`output/pdf/sestina_v10.pdf` and `.log` are the revised build.

## 6. Unresolved author queries

1. **§6.4, forced endgame.** The clause "on a lower threshold once $\pi_{\rm team} > 1-5\times10^{-4}$" was removed: the only such test (`v05.hpp:1123`) is on the voluntary path and unreachable in the frozen configuration. Was it describing v0.5's threshold mode? If it should be documented, attribute it.
2. **§9.4.1 / T03.** If a ranking-preservation proposition for the sequential conditioning is intended, supply it with assumptions and proof; otherwise the empirical statement stands. Consider having the generator emit the count of distinct declarations behind `\vsevenLoneDecls` (the recorded 210,804 includes re-runs of the same seeds).
3. **§9.4.5 / T15(d).** `docs/v07/ADVERSARIES.md` (summary A7 and §4F) states that a construction raising the target's forced-endgame incidence was measured at −17 pp; no artifact row supports it, and the planned arm X07 (objective `forced`) has no result. The paper keeps its E.2 correction ("no cell behind it"); the phase-2 document still carries the figure. Which record is right?
4. **Appendix G, repository.** The cited URL (`github.com/dylann29/fish-optimization`) differs from the git remote (`github.com/dylann4500/FishLab.git`). Which is canonical? The annotated tag `sestina-v1.0` is now cited; confirm whether the revised paper gets a new tag.
5. **Appendix F.** D24 follows D27 in `docs/v07/FINAL-RESULTS.md` with no explanation; a one-line note there would let the appendix inherit it.
6. **Appendix G, environment.** Compiler version and machine load during the battery are not recorded; the research log's "macOS 25.5.0" is a Darwin kernel version.
7. **§1 / §3.1.** The "no prior computational work" claim is now scoped to the project's searches but gives no dates or index list; supply them if available.
8. **Bibliography.** The new `nelo-fishtest` entry has no URL or access date (none was in the vetted source).
9. **§7.9 / T06.** The text says a planted effect "below about a point and a half" was not recovered, with the bracketing explained. The directly supported statement is that the 0.88 rung was not recovered; the author may prefer that form.
10. **Appendix C / T12.** The S3, S4, S6 and S7 `PASS` entries are literal text, not generator output; consider adding them to `verdicts()`. The phase-5 record's condition-7 clause ("every control behaves as specified") still names no reading.
11. **§11.9 / T10.** No control cell bounds the cross-deal residual's effect on a scored win rate; a fresh-versus-reused strength cell would. This is additional research, listed as future work.
12. **§7.3 / T04.** A joint bootstrap of between-cell contrasts needs per-deal outcomes, which the scored artifacts do not retain; listed as future work.
13. **`docs/RELEASE-v1.0.md`** states 72 pages and describes the previous PDF; it was left as a historical record. Decide whether to update it for the revised build.
14. **§2.2 / T15.** The transfer-channel oracle was measured on v0.4 mirror self-play and the arbitration oracle on three of five seeds; both are now stated. A SESTINA-era re-measurement would be new work.
15. **§3.5 and §2.3.** The A-loss-recall claim and the perfect-information "degeneracy" claim were narrowed to what is established; if derivations exist, restore them with conditions.
16. **Title.** The versioned title was kept; the audit offered an unversioned alternative.

## 7. Not done

- Audit §7.2 (second-pass condensation) and the BibTeX/bibliography-style conversion.
- Full normalisation of venue names, author formatting, DOIs and access dates across
  all 71 entries; only the audit's specific items were changed.
- Any rerun of experiments, sensitivity reruns with a corrected implementation, or joint
  bootstraps: these are research tasks recorded as future work.


## 8. Second pass: Astra independent review (21 September 2026)

The independent post-revision review (`SESTINA-v1.0-Astra-independent-review.md`) was implemented
as a second pass on the same branch. The finding-by-finding plan and disposition are in
[SESTINA-v1.0-ASTRA-DISPOSITION.md](SESTINA-v1.0-ASTRA-DISPOSITION.md). In summary:

- **Unresolved, disclosed:** A1, C1 and Q11 require a scored rerun with agent state reset between
  deals (or a dependence-aware analysis of retained worker sequences). No such run exists. Every
  interval in the paper is now labelled nominal, adjacent to its first presentation, in the
  abstract and in the conclusion, and the primary result is stated as an observed edge under the
  recorded implementation on which the registered rule was satisfied.
- **Done:** A2–A10, B1–B5, B8–B10, C2–C7, and the open-decision and author-query items listed in
  the disposition, including the unversioned title, the search-method note, the probe-backed
  clairvoyance statement, the corrected repository URL, generator-emitted declaration counts, and
  the correction note appended to the phase-2 adversary record.
- **Partial:** B6 (two web sources could not be re-fetched; full stylistic normalisation not
  attempted) and B7 (conservative condensation applied; page count unchanged at 82 because the
  required disclosures added as much as the compression removed; a deeper cut is an author
  decision with a concrete proposal in the disposition).
- **Supporting metadata:** `docs/RELEASE-v1.0.md` carries a dated revision-history note above its
  original text; `docs/v07/ADVERSARIES.md` carries a dated correction note after its original text;
  the annotated tag `sestina-v1.0-paper-r1` marks the build described by the revised paper
  (created locally; pushing is the author's action).
