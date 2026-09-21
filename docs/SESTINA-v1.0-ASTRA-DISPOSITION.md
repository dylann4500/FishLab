# SESTINA v1.0: second-pass plan and disposition (Astra independent review, 21 September 2026)

**Input:** `SESTINA-v1.0-Astra-independent-review.md`, treated as the authoritative second-pass
editorial report. **Baseline:** branch `paper/sestina-v10-editorial-revision` at commit `8d1decd`
(first-pass revision, 82 pages).

Part 1 is the implementation plan written before any edit: each finding mapped to the manuscript
location(s) and the proposed edit. Part 2 is the disposition after implementation. Findings that
need a new experiment are marked **unresolved** and no result is invented for them.

Locations are section numbers of the revised paper and the modular source file
(`paper/sections_v07/…`). "Generator" means `engine/build_tables_v07.py`.

## Part 1: implementation plan

### A. Errors introduced by, or remaining in, the revised presentation

| ID | Severity | Location | Proposed edit |
|---|---|---|---|
| A1 | blocking (new experiment needed) | §7.3 "Per-bank intervals" (07-evaluation.tex); §11.1 first interval (11-results.tex); abstract; §13; §11.9; §12.6 | State adjacent to the first interval, in the abstract and in §13, that the deal-block bootstrap treats deals as independent resampling units, that the scored implementation's cross-deal state (§6.8) violates that assumption for searching agents, and that every reported interval is therefore **nominal**; present the primary result as an observed edge under the recorded implementation and the registered rule as satisfied on the nominal interval. Add a canonical statement in §12.6 that coverage is not established under the actual construction and that a corrected fresh-state rerun (or a dependence-aware analysis of worker-ordered sequences) is required for a coverage-validated claim. No rerun is available: the finding stays **unresolved**. |
| A2 | should fix | §2.3 "Decentralised team decision-making" (02-game.tex) | Replace "the coordinator's information state is exactly the belief object the engine already maintains" with the reviewer's sentence distinguishing the common-information belief over joint private states from the seat-conditioned belief. |
| A3 | should fix | §6.2 definition of ρ and $\mathcal P_t$ (06-agent.tex); notation table row for $\mathcal P_t$ (05-inference.tex) | Define $\mathcal P_t$ as a partial public-event digest over each event's kind, actor, target, card and outcome, state that it omits the half-suit index and a declaration's named allocation, and state that whether those omissions can matter to a tie case was not established. |
| A4 | should fix | §2.2 arbitration sentence (02-game.tex); change-log T15 row | Replace "bound the value of choosing among a fixed set of proposals" with "measure the effects of two diagnostic arbitration interventions over the fixed proposal set"; reclassify T15's arbitration claim as unsupported as a game-value bound, measurement confirmed. |
| A5 | should fix | §5.2 replay sentence (05-inference.tex) | Use the reviewer's wording: a replay failure indicates dependence not represented in the replay construction; source inspection localised the observed residual to persistent rollout state; the carrier field and the scored effect were not isolated. |
| A6 | should fix | §3.3 "three uncorrelated opponent seats" (03-related.tex) | Replace with "three opponent seats with distinct private information at the leaf; the cited two-player continuation construction was not instantiated for this setting". |
| A7 | should fix | §7.7 "unavailable … by construction" (07-evaluation.tex) | Say the named holdout banks were guarded against routine match access, digests were committed in advance, the guard is an environment-variable override and some drivers bypass it (D5/D6); do not claim unavailability by construction. |
| A8 | minor | §2.3 "Planning limitations" (02-game.tex) | Change the second "Second" to "Third". |
| A9 | minor | Table 20 header (generator `b8_dialects`) | Header "Dialect" → "Rule variant"; regenerate. |
| A10 | minor | 06-agent.tex:127, 07-evaluation.tex:156, G-reproduction.tex:26, E-corrections.tex:25,96,107 | Remove boldface from the six whole-sentence emphases; keep bold for defined terms and labels. |

### B. Audit recommendations missed or incompletely implemented

| ID | Severity | Location | Proposed edit |
|---|---|---|---|
| B1 | should fix | §7.6 "cleared context" (07-evaluation.tex); §4.4 cross-reference | Replace with the concrete workflow recorded in `docs/v07/PHASE-PROMPTS.md` and the preregistration: each phase was a separate automated agent session (a large language model operating the repository under the author's supervision), started without conversation history from earlier phases, instructed to read the preregistration and not the development documents or training logs; the restriction was by instruction, the session had read access to the repository, and the same author configured every phase, so no evaluator independence is claimed. |
| B2 | should fix | §3.1 "Computational prior work" (03-related.tex); §1 (01-introduction.tex) | Add a compact search-method note from the records: survey of 2026-08-21 (arXiv API queries `all:"Canadian Fish"`, `all:"Literature card game"`, `abs:"Go Fish"`, `all:"half-suit"`; GitHub code search over `literature card game`, `canadian fish card game`, `fish card game ai`, `half suit card game`; RLCard's environment list; Google Play), refreshed 2026-08-22 and 2026-08-23; records `research/v04/lit/fish-prior-art.md`, `research/v05/lit/v05-refresh.md`, `research/v06/notes/R7-literature-search.md`. Use "we did not identify" throughout. |
| B3 | should fix | §9.3 "channel ceilings", "measured ceiling"; §9.4.1 "oracle ceiling" (09-development.tex) | Replace with "diagnostic intervention estimates under the evaluated condition", naming the conditioning (responder class, budget, handicap, banks, implementation). |
| B4 | should fix | §6.7 "every action would score alike" (06-agent.tex); §2.3 and §3.2 clairvoyance wording | One scoped statement, supported by the recorded omniscient-play probe (`research/v06/notes/R8-game-structure.md`: with both teams omniscient, over 300 mirror games the non-moving team's score equalled its deal-time locked count in every game): under perfect information the implemented rules give the team on turn every half-suit not dealt outright to the opponents, so a clairvoyant rollout assigns nearly the same value to every action; this is an implementation-level result verified by that probe, not a formal value bound. Add a transcribed macro for the probe's game count. |
| B5 | should fix | §12.7 "cannot be ported" (12-discussion.tex) | "A direct port is not justified without specifying and calibrating the continuation policy." |
| B6 | should fix | bibliography.tex | Give `nelo-fishtest` its stable URL (Stockfish docs, "Statistical Methods and Algorithms in Fishtest") and access date; add access dates for the web sources fetched during this pass (pagat, dorsa, somani, somani-server, wikiliterature, strategy-site, cantate.be); correct `somani` to "2019--" per the file's own header (creation 2019-06-10); normalise the kubicek2025lamir venue/year form; state in the header which sources could not be re-fetched (Play Store, archive.org). |
| B7 | should fix (conservative pass) | Whole manuscript | Conservative compression only, preserving every disclosure: (1) shorten the generated deviation register's elision from 460 to 240 characters (source document carries the full text); (2) compress Appendix E entries and Appendix G's artifact list and duplicated provenance paragraph; (3) trim §4 narrative; (4) state the cross-deal limitation once canonically in §12.6 and cross-reference elsewhere (A1 requires short adjacent disclosures, which stay). A cut to the reviewer's suggested range would require dropping tables or appendix records; that is listed as an author decision with a concrete proposal. |
| B8 | minor | §3.3 full-solver sentence (03-related.tex) | Name the classes and qualify with "typically", stating the requirement for the cited methods. |
| B9 | minor | §12.5 (12-discussion.tex) | Use the reviewer's sentence: the study relies on the listed internal validation measures, which do not substitute for external replication. |
| B10 | should fix | Appendix G "Availability" (G-reproduction.tex) | Correct the repository URL to the public remote (`https://github.com/dylann4500/FishLab`; the cited `dylann29/fish-optimization` returns 404 and the former name `dylann4500/fishbot` redirects to FishLab); keep `sestina-v1.0` as the release tag of the original build; create and cite an immutable revision tag `sestina-v1.0-paper-r1` on the final commit of this pass (local; pushing is the author's action). |

### C. Remaining overstatements and inconsistencies

| ID | Severity | Location | Proposed edit |
|---|---|---|---|
| C1 | blocking (new experiment needed) | abstract; §11.1; §13 | As A1: nominal-interval qualification carried into the abstract and conclusion; "documented improvement" → "observed improvement under the recorded implementation, with nominal intervals". **Unresolved** pending a corrected run. |
| C2 | should fix | §7.7 calibration sentence (07-evaluation.tex) | Scope the null interpretation to the registered responder class, budget, handicap family and banks. |
| C3 | should fix | §7.9 and §9.2, E.1 (07-evaluation.tex, 09-development.tex, E-corrections.tex) | Exact rung wording: the 0.88 pp rung was not recovered and the 1.53 pp rung was; add transcribed macro `\vsevenFloorRungBelow{0.88}` (source `docs/v07/ADVERSARIES.md`). |
| C4 | should fix | §11.7 (11-results.tex); §12.6 | "The residual estimate is not statistically interpretable under the current propagation, and no separate ladder intervention was conducted." |
| C5 | should fix | §11.9 (11-results.tex) | "certifying condition" → "registered test condition". |
| C6 | should fix | §2.3 (02-game.tex); §1; §3.1 | "Within the search scope described in §3.1, we did not identify prior work that addresses this case." |
| C7 | should fix | §9.3, §11.8 caption, §2.3, §12.4 (09/11/02/12) ; cross-play "independently trained" (§1, §11.6, §12.1, §12.6) | "separately fitted adversarial-search classes using common cross-entropy-method infrastructure", naming the shared code and objectives; cross-play runs "separately trained on disjoint fitting banks with the same fitting infrastructure". |

### D. Open decisions and author queries

| Item | Decision | Edit |
|---|---|---|
| D1.1 / Q16 | adopt | Title → "SESTINA: Constraint-Based Inference and Empirical Evaluation in Six-Player Canadian Fish"; `pdftitle` likewise; "v1.0" stays in the abstract, the agent's configuration name (`\sestina`), the running head, artifact metadata and the release record. |
| D1.2 | adopt (partial) | B7 conservative pass as above. |
| D1.3 | adopt | keep `tocdepth=2`. |
| D1.4 | adopt | Present-tense prose, headings and captions use "behavioural validity gate"; "soundness gate" only as the protocol's quoted name. |
| D1.5 | adopt | Appendix B: one sentence distinguishing the tabulated weights $w$ from the scale $\lambda$ and the value term. |
| D1.6 / Q8 | adopt | Stable URL and access date found for the second Fishtest source; entry retained. |
| D1.7 / Q9 | adopt | C3 wording. |
| D1.8 | adopt | no change to Eq. (24). |
| D1.9 / Q1 | adopt | clause stays removed. |
| D1.10 | adopt | Figure 3 unchanged. |
| D1.11 / Q10 | adopt | Column "Recorded" → "Recorded verdict" in Table 11 and Appendix C; PASS entries stay source-attributed text. |
| D1.12 / Q12 | adopt | no cross-play interval. |
| D1.13 | adopt | §9.1 states that the protocol's term "frontier" denotes a comparison set. |
| D1.14 | adopt | E.2 presents the bibliography check as the author's internal correction inventory. |
| D1.15 | adopt | transcribed macros retained; Q2 counts moved to generator output. |
| D1.16 | adopt | rule additions stay labelled as the evaluated variant. |
| Q2 | adopt | Generator emits the recorded total (sum of `metrics.declShipAccuracy.n` over the three K2 capture files) and the distinct count (one cell per distinct seed and urgency setting); §6.4, §9.4.1 and E.1 quote both with their definitions. |
| Q3 | adopt | Append a dated correction note to `docs/v07/ADVERSARIES.md` stating that no artifact supports the −17 pp figure; the original text is left in place. |
| Q4 | adopt (partial) | URL corrected on evidence; revision tag created locally; pushing and any GitHub release are the author's actions. |
| Q5 | keep | Appendix F already states the ordering; no chronology exists in the record to add. |
| Q6 | adopt | Appendix G: "under macOS (the record gives the Darwin kernel version …)", via a transcribed macro. |
| Q7 | adopt | B2. |
| Q11 | **unresolved** | Corrected fresh-state scored rerun not available; A1/C1 disclosures added; listed first in future work. |
| Q13 | adopt | `docs/RELEASE-v1.0.md`: add a dated revision-history note above the original text; the original text is left intact as the record of the v1.0 release. |
| Q14 | adopt | "bound" removed (A4); no remeasurement. |
| Q15 | adopt | B4, B5. |

### Verification-classification adjustments

T09 → implementation completed by A3; T10 → consequence extended to interval coverage (A1); T15 →
arbitration claim reclassified (A4), urgency figures framed as condition-specific (B3); T19 →
completed by B2 and B6 to the extent the sources could be verified; T20 → completed by A7 and B10.

### Final consistency search

After the edits: `bound`, `ceiling`, `certif`, `unavailab`, `uncorrelated`, `clairvoyan`,
`independen`, `v1.0`, checked occurrence by occurrence against the narrower formulations.

## Part 2: disposition after implementation

Status values: **done** (implemented as planned), **partial** (implemented in part; the remainder
is stated), **unresolved** (requires a new experiment or an author action; no result invented),
**no change** (decision adopted without an edit).

### A

| ID | Status | Where it landed | Notes |
|---|---|---|---|
| A1 | **unresolved** (disclosed) | §7.3 new paragraph "Independence assumption"; §11.1 first paragraph; abstract; §13; §11.9 last paragraph; §12.6 "Cross-deal state persistence" (canonical statement); §12.7 first item | Every interval is now labelled nominal, the primary result is stated as an observed edge under the recorded implementation, and the registered rule is described as satisfied on the nominal interval. No corrected run exists, so coverage remains unestablished. |
| A2 | done | §2.3 "Decentralised team decision-making" | reviewer's sentence used |
| A3 | done | §6.2 definition of $\rho$; notation table row for $\mathcal P_t$ | hashed fields named; omissions named; relevance to tie cases stated as not established |
| A4 | done | §2.2; change log T15 row | "measure the effects of two diagnostic arbitration interventions"; T15 arbitration claim reclassified as unsupported as a game-value bound, measurement confirmed |
| A5 | done | §5.2 | reviewer's wording |
| A6 | done | §3.3 | reviewer's wording |
| A7 | done | §7.7 | guarded procedure with prior commitments; overrides and bypasses named (D5, D6) |
| A8 | done | §2.3 | "Third" |
| A9 | done | Table 20 header via generator `b8_dialects` | regenerated |
| A10 | done | 06-agent, 07-evaluation, G-reproduction, E-corrections (three) | six whole-sentence emphases removed; bold now marks only defined terms, table labels and run-in headings |

### B

| ID | Status | Where it landed | Notes |
|---|---|---|---|
| B1 | done (author check recommended) | §7.6; §4.4 cross-reference | The workflow is now described from `docs/v07/PHASE-PROMPTS.md` and the preregistration: separate automated agent sessions under the author's supervision, restriction by instruction, read access to the repository, same author throughout, no evaluator independence claimed. This is a disclosure the author may wish to word differently. |
| B2 | done | §3.1; §1 | search dates (21, 22, 23 August 2026), indexes, query families and record files stated; "we did not identify" throughout |
| B3 | done | §9.3 (two places); §9.4.1 | "diagnostic intervention estimate under the evaluated condition", conditioning named |
| B4 | done | §2.3; §6.7; §4.3; §3.2 already scoped | one probe-backed statement (both-teams-omniscient probe, 300 mirror games, identity in every game; new source-attributed macro `\vsevenOmniProbeGames`); "not a formal value bound" |
| B5 | done | §12.7 | reviewer's sentence |
| B6 | partial | bibliography | `nelo-fishtest` now has the Stockfish documentation URL and an access date; access dates added for the seven web sources re-fetched on 21 September 2026; `somani` start year corrected to 2019 per the file's own header; `kubicek2025lamir` normalised. Not done: the Play Store listing and the archive.org snapshot could not be re-fetched from the editing environment and are unchanged; a full stylistic normalisation of all 72 entries was not attempted beyond the items the two reviews named. |
| B7 | partial | Appendix E (five entries tightened), Appendix G (artifact list and duplicated provenance paragraph compressed), §4.3 (repeated explanation removed), Appendix F (generated elision shortened from 460 to 240 characters) | Page count is unchanged at 82: the compression was offset by the disclosures A1/C1, B1, B2 and B4 required. Reaching the reviewer's suggested range needs decisions the editor should not take alone: moving Appendix D (full panel, 3 pages) and Appendix F (deviation register, 3 pages) to the repository with an index; merging §7 and §10; cutting §3 to about three pages; replacing §4 with a version table. Estimated result 60–65 pages. |
| B8 | done | §3.3 | classes named, "typically" |
| B9 | done | §12.5 | reviewer's sentence |
| B10 | done (push pending) | Appendix G "Availability" | URL corrected to `https://github.com/dylann4500/FishLab` on evidence (the cited URL returns 404; the former name redirects); `sestina-v1.0` untouched; annotated tag `sestina-v1.0-paper-r1` created locally on the final commit of this pass. Pushing the branch and the tag is the author's action. |

### C

| ID | Status | Where it landed | Notes |
|---|---|---|---|
| C1 | **unresolved** (disclosed) | abstract; §11.1; §13 | "observed improvement under the recorded implementation, with nominal rather than coverage-validated uncertainty"; carried into abstract and conclusion |
| C2 | done | §7.7 | scoped to the registered responder class, budget, handicap family and banks |
| C3 | done | §7.9 (two places); §9.2; E.1 | exact two-rung wording; new source-attributed macro `\vsevenFloorRungBelow{0.88}` |
| C4 | done | §11.7 | reviewer's sentence |
| C5 | done | §11.9 | "registered test condition" |
| C6 | done | §2.3; §1; §3.1 | "Within the search scope described in §3.1, we did not identify …" |
| C7 | done | §11.8 caption; §2.3; §9.3; cross-play: RQ2, Table 17/18 captions, §12.1, §12.6; Appendix B | "separately fitted … using common cross-entropy-method fitting infrastructure (the `tune` driver)"; cross-play runs "trained separately with the same fitting infrastructure on disjoint fitting banks" |

### D

| Item | Status | Notes |
|---|---|---|
| D1.1 / Q16 | done | Title and `pdftitle` unversioned. The configuration name `SESTINA v1.0` (macro `\sestina`) is retained in the abstract, prose, running head and generated tables as the identifier of the frozen configuration. |
| D1.2 | partial | see B7 |
| D1.3 | no change | `tocdepth=2` kept |
| D1.4 | done | present-tense prose, Table 10 caption and Table 11 row use "behavioural validity gate"; "soundness gate" appears only as the protocol's quoted name |
| D1.5 | done | Appendix B sentence on $w$, $\lambda$ and $\beta$ |
| D1.6 / Q8 | done | see B6 |
| D1.7 / Q9 | done | see C3 |
| D1.8 | no change | Eq. (24) restriction kept |
| D1.9 / Q1 | no change | clause stays removed |
| D1.10 | no change | Figure 3 unchanged |
| D1.11 / Q10 | done | columns renamed "Recorded verdict" and "Reading with uncertainty" in Table 11 and Appendix C; PASS entries remain source-attributed text, as the appendix states |
| D1.12 / Q12 | no change | no cross-play interval |
| D1.13 | done | §9.1: the protocol's term denotes a comparison set |
| D1.14 | done | E.2 presents the bibliography check as the author's internal inventory |
| D1.15 | done | transcribed macros retained; Q2 counts are now generator output |
| D1.16 | no change | rule additions remain labelled as the evaluated variant |
| Q2 | done | generator emits `\vsevenLoneDeclsTotal` (210,804, every recorded declaration in the three capture files) and `\vsevenLoneDeclsDistinct` (125,534, one cell per distinct bank and urgency setting); §6.4, §9.4.1 and E.1 quote both with their definitions |
| Q3 | done | dated correction note appended to `docs/v07/ADVERSARIES.md`; original text untouched |
| Q4 | partial | see B10 |
| Q5 | no change | Appendix F already states the ordering; the record contains no chronology to add |
| Q6 | done | Appendix G names the Darwin kernel version via a source-attributed macro and does not call it a macOS version |
| Q7 | done | see B2 |
| Q11 | **unresolved** | corrected fresh-state scored rerun not available; A1/C1 disclosures added; listed first in §12.7 and §13 |
| Q13 | done | `docs/RELEASE-v1.0.md`: dated revision-history note inserted above the original text, which is left intact |
| Q14 | done | "bound" removed (A4); no remeasurement |
| Q15 | done | B4, B5 |

### Verification-classification adjustments

Applied to the change log's T-item table: T09 completed (A3); T10 consequence extended to interval
coverage (A1); T15 arbitration claim reclassified, urgency figures condition-specific (A4, B3);
T19 completed to the extent the sources could be verified (B2, B6); T20 completed (A7, B10).

### Consistency search (after the edits)

`bound`: remaining uses are lower/upper bounds of intervals, the search guard's lower confidence
bound, and negated statements ("not a bound"). `ceiling`: only inside macro names. `certif`:
"certificate" (the defined constraint object), the protocol's own label "CERTIFIED", and a negated
"not itself a certified bound". `unavailab`: the removed claim's negation in §7.7 and two unrelated
uses. `uncorrelated`: none. `clairvoyan`: every use is now scoped to the implemented rules under
omniscient play or to the greedy diagnostic chooser. `independen`: statistical independence,
"separately fitted/trained" where infrastructure is shared, and "independently authored/developed"
for future external work. `v1.0`: the configuration name, running head, release record and
artifact identifiers; the title is unversioned.

### Checks executed in this pass

| Check | Result |
|---|---|
| `tectonic` build of `paper/sestina_v10.tex` | exit 0, 82 pages, 0 overfull boxes, 0 undefined references or citations |
| flattened standalone regenerated with `paper/inline.py` and built | exit 0, 82 pages |
| `engine/build_tables_v07.py --paper` | 622 macros, 27 tables; new macros `\vsevenLoneDeclsTotal`, `\vsevenLoneDeclsDistinct`; Table 20 header; deviation-register elision |
| `paper/check_provenance.py --version v07` | passes |
| web verification (21 September 2026) | `github.com/dylann29/fish-optimization` 404; `github.com/dylann4500/FishLab` public, default branch `main`; `dylann4500/fishbot` redirects to FishLab; Stockfish documentation page for the Fishtest statistics located; pagat, dorsa, somani, somani-server, wikiliterature, strategy-site and the cantate.be PDF reachable; Play Store and archive.org not reachable from the editing environment |
| rendered pages inspected | 1, 9, 30, 45, 59, 61, 80 |

No experiment was rerun, no engine source was changed, no artifact was modified, and no
measurement, hash, verdict label or historical quotation was altered.


## Part 3: round-2 verification (Astra, 21 September 2026)

Astra's implementation check found nothing blocking provisional circulation and listed ten
corrections plus status changes. All ten were applied on the same branch; the statuses below
supersede Part 2 where they differ.

| Finding | Status | Where it landed |
|---|---|---|
| F1 headline interval called a bootstrap interval | done | abstract, §11.1, §13: "nominal pooled 95% interval formed by normal approximation from the per-bank deal-bootstrap intervals"; endpoints unchanged |
| F2 residual inferential wording | done | Table 11 rows 1 and 5 and Appendix C (primary, S3, S6): "passes the registered rule using nominal intervals; coverage unestablished"; Table 18 caption rewritten (observed separation, not a coverage-validated test; distinct vectors stated separately); §11.8 "registered numerical criterion is satisfied using the nominal intervals"; §7.1 convention and the captions of Figure 3 and Tables 8, 12, 20, 23 and §11.2.1 now say "nominal" |
| F3 rerun said to validate old intervals; "violates" too categorical | done | §7.3, §11.9, §12.6: independence and coverage "have not been established for comparisons involving searching agents", non-searching cells unaffected, "nominal" as a labelling convention; §12.6 and §12.7: a fresh-state rerun produces new estimates whose validity still depends on their constructions, per-deal capture and joint paired contrasts required, existing results remain measurements of the original implementation; §13 aligned |
| F4 clairvoyance action-level conclusion | done | §2.3 reports the probe descriptively (terminal-score identity in 300 mirror games; motivates concern about weak discrimination; does not establish action values; no action-level evaluation recorded); §3.2, §4.3, §6.7 aligned |
| F5 observer-conditioned count | done | §3.3 separates the one-seat-conditioned count from the public-history support and states the cardinality as scale, not an algorithmic bound |
| F6 tag availability | done (publication still pending) | Appendix G states that the branch is published, that the tag marks its final commit, and that the tag's publication was pending when the revision was built; the local tag is re-pointed at the final commit of this pass |
| F7 ladder bracketing | done | §7.9: two observed recovery outcomes; no continuous threshold estimated; 1.53 pp a historical choice |
| F8 workflow and search disclosures | done editorially; **author confirmation requested** | §7.6 distinguishes what the phase instructions required from what the execution records document (instructions and outputs, not compliance); §3.1's search note is drawn from the three record files, which exist in the repository but were not in the review package |
| F9 count and cross-reference | done | §2.3 "Three considerations"; the cross-deal reset paragraph is now the first item of §12.7 and §12.6 says "first priority" |
| F10 residual uncertainty wording | done | §12.6 "has only incidental observations, not a quantified bound"; §11.7 "would be wider if the relevant covariance were non-negative"; Appendix E "both propagated ranges include zero, but neither supports a calibrated inferential conclusion" |

Status changes accepted from the review: A1 and C1 remain **unresolved**, with the disclosure now
scoped as the review asked; A8 done; B1 and B2 editorially implemented with source verification
pending on the author's side; B4 done as scoped; B8 done; B10 **partial** until the tag is
published; C3, C4, D1.11 done; Q11 unresolved with the rerun described correctly and placed first in
future work; T10 disclosed with the scope corrections; T19 and T20 partial. B6 and B7 remain
partial; on B7, the generated deviation register now elides at a sentence boundary rather than at a
character count, so every printed entry ends with a complete sentence, and the page count is 83.

Checks: build exit 0, 83 pages, 0 overfull boxes, 0 undefined references; provenance audit passes;
standalone regenerated and built; pages 1, 60 and 78 inspected. No measurement, artifact, hash,
verdict label or quotation changed.
