# KRAKEN v1.1 — drop-in integration

One bot. One entry point. No knobs.

This directory is everything another project needs to play **KRAKEN
v1.1** (the engine this project called KV's FishBot v0.6 until 2026-08-28)
on its own site or in its own engine, without reading the research
code. It is the mirror of the bridge this project built to run
[dylann4500/fishbot](https://github.com/dylann4500/fishbot) v0.7 inside our
engine, and it works the same way: the host owns the game, the bot answers
one question at a time.

## What the bot is

| | |
|---|---|
| name | **KRAKEN v1.1** |
| configuration | `opponent_gamma 0.35 · n_draws 480 · lookahead 0.25 / depth 3 / beam 4 · endgame_m 0 · claim_forced_exhaustive 1` |
| defined in | `fish4/registry4.py` as `V06_DEPLOYED` (the self-test asserts this file matches it) |
| requires | Python 3.11+, `numpy`. Nothing else. No network, no model files, no GPU. |

There is exactly one configuration. It is what plays on our site and what
produced every number below; there is no "strong mode" or second variant to
pick between.

## Quick start

```bash
python -m kraken.decide --self-test     # ~30 s, proves the door works
echo '{"op":"version"}' | python -m kraken.decide
```

Then drive it: one JSON object per line in, one per line out. Keep the
process alive for a whole match (fast) or send a single line and close the
pipe (simple) — it is stateless either way, so a lost or repeated request
cannot corrupt anything.

```bash
echo '{"seat":0,"turn":0,
       "hand":["2C","3C","4C","2H","3H","9D","TD","JD","QD"],
       "hand_counts":[9,9,9,9,9,9],
       "set_winner":[null,null,null,null,null,null,null,null,null],
       "history":[]}' | python -m kraken.decide
# {"action": {"type": "ask", "target": 1, "card": "5C"}}
```

`python -m kraken.decide` speaks the protocol documented in full at the
top of `decide.py` — request fields, the three action shapes, the event
shapes, and the seeding rule. Two conveniences worth knowing:

- `{"op":"cards"}` returns the card-name table and the nine half-suits, so
  you never have to reimplement our card ordering to talk to us.
- `{"seed": N}` makes a decision reproducible; omit it and the seed is
  derived from the history, so a match is still replayable from its log.

## Three things a host MUST get right

These are not style preferences. Each one, gotten wrong, makes this bot
look weak rather than broken — which is the worst kind of bug, because
nothing errors and the games just quietly go the other way. We learned all
three the hard way when our bot was first run inside another engine's
server and lost games it wins here.

**1. Poll every seat off-turn, if your rules allow off-turn declarations.**
v0.7's engine sets `outOfTurnDeclare = true` by default: any seat may
declare the moment it knows a half-suit. If you only ever ask the seat whose
turn it is, our seats wait their turn while yours race — and whoever
declares first takes the set. Measured over 240 games on identical deals
(`scripts4/dialect_gap.py`):

| arbiter | our margin |
|---|---|
| nobody declares off-turn | **+2.675** sets/game |
| both sides may (your dialect, played fairly) | **+2.375** sets/game |
| only *your* side may (a host that never polls us) | **+1.575** sets/game |

Those three absolutes were measured through bridge revision 1 (see *One number
we took back* below), so read them as ±0.08. What the table rests on is the
**gaps** between the arms, and those are unaffected — all three ran through
the same bridge, so a defect common to all three cancels out of the
comparison. The point stands at full strength: a host that never polls us
costs us about 0.8 sets a game.

So poll us:

```json
{"op":"offturn", "seat":3, "turn":0, "hand":[...], "hand_counts":[...],
 "set_winner":[...], "history":[...]}
```

You get `{"action": {...declare...}}` or `{"action": null}`. It answers with
a declaration **only** when the public record alone pins every card of a
half-suit, so it is always correct and never a gamble — poll it for all six
seats between moves and drop the ones that answer null.

**2. Cards are NAMES, never integers.** `"2C"`, `"TD"`, `"AH"`, `"BJ"`,
`"RJ"`. We deliberately **reject** integer card ids with an error. Your
engine numbers cards `set*6+idx` over a different permutation of the sets
than ours, so an integer that means one card to you means another to us —
and a bot playing a scrambled hand plays legally, badly, and silently. If
you need our ordering for any reason, ask `{"op":"cards"}`.

**3. Never treat an `{"error": ...}` response as a move.** If a request is
malformed we say so rather than guessing. A host that catches the error and
substitutes a random legal move will lose every game and show you nothing.
Log it and fix the request; `--self-test` will tell you if the problem is on
our side.

## The rules it assumes

Standard six-player, 54-card Literature: nine half-suits of six, teams by
seat parity ({0,2,4} vs {1,3,5}), ask only what you can legally ask, and
**any wrong declaration awards the half-suit to the opposing team** —
matching v0.7's native rule. Set it explicitly if you like; the default is
already this. The legacy void variant (`"wrong_distribution_outcome":
"null"`, where a right-team wrong-split declaration scores for nobody) is
still accepted per request because this project's older published results
were measured under it — do not use it for new games, it is not the game.

One rule of yours we do not implement: `cardlessMayDeclare`. A seat of ours
with no cards will answer the off-turn poll normally (deduction does not
need cards), but will not be dealt into the on-turn loop. If that matters
for your arbiter, tell us and we will extend the protocol rather than have
you paper over it.

## Verifying the integration, not trusting it

`--self-test` replays three complete games and compares **every** decision
made through the JSON protocol against the same decision made by importing
the agent directly, with the same seed. It currently reports **329
decisions compared, 0 mismatches**, and it also exercises the card table,
the version handshake, and a malformed request.

That is the check that matters for an integration: not "does the bot play
well" but "is this door the same bot". If you change anything in this
directory, run it again.

## How strong it is

Measured, pre-registered, and reproducible from this repository. Every
figure below is a paired duplicate-deal comparison — identical deals played
by both sides, seats rotated — under the standard misdeclaration rule.

| against | result |
|---|---|
| **FishBot v0.7** (frozen `v07:r12=25…`, its own C++ engine, 10,000 games) | **+2.347** [+2.293, +2.400] sets/game, **63.0% of decided sets**, 80.4% of games won, zero substituted moves |
| this project's own v0.3 champion, 500 pairs | **+2.412** [+2.088, +2.736] sets/pair |

The full method, the pre-registrations, and the results files behind these
are in `paper/fishbot_v06.pdf` and `prereg/`.

Where the margin comes from is worth knowing before you run it, because it
is not where people expect. The two engines find cards at similar rates —
51.8% of asks hit against 48.1%. They differ on knowing when a half-suit is
**finished**: this bot declares correctly 96.5% of the time against 78.9%.
Card-reading is the visible skill; knowing when to stop reading is the one
that scores.

### One number we took back

An earlier version of this table said **+2.54 sets/game, 64.3%**, from a
1,000-game run, and a 10,000-game run here reported +2.425. Those are
withdrawn — the replacement re-plays the same 10,000 deals, so it is a
like-for-like correction rather than a differently-powered second opinion.

Reading v0.7's C++ adversarially turned up a defect not in either
engine but in **our** bridge into his, and it ran in our favour: forced to
declare, his own driver picks the first live half-suit the mover holds a card
in (`engine/src/game.hpp:535`), and our bridge passed the first live
half-suit, full stop. His `bestGuess` was therefore sometimes asked to name
all six owners of a half-suit it held nothing of — a question his own driver
never asks him, which he gets wrong nearly every time, and which under the
opponent-award rule hands us the set.

Priced paired on all 10,000 deals replayed under both bridge revisions: worth
**−0.0784** [−0.0881, −0.0687] sets/game to us. Real, about 3% of the margin,
and an order of magnitude smaller than the first estimate off 40 games. His
declaration accuracy rises 76.2% → 78.9% under the repair; ours moves 96.7%
→ 96.5%, which is the control. The table above is the repaired number, and
`results/BRIDGE_REVISIONS.md` says which stored journal is which.

We mention it because you are being asked to run this bot against your own,
and a margin measured through our apparatus is a claim about our apparatus
too. If you find another one, we would rather hear it than not.

## What we tried from your side and could not make work

Honesty is cheaper than a surprise later. We ported v0.7's contestation
term — the coordinate your attribution study credits with essentially that
engine's whole cycle gain — into our ask objective and swept it in both
directions over 4,000 games against v0.7. **Every dose lost**, the largest
by −1.12 sets/game, monotone in the dose. We do not think that refutes it;
it plainly works where it was fitted. We think an ask-scoring coordinate
tuned against one belief representation is not portable between engines,
which is a result neither project could have obtained alone. It is written
up in §21 of our paper, with the table.

## Licence and attribution

Research code, published for interoperability and comparison. If you run it,
please call it **KRAKEN v1.1** so results stay attributable to a
specific configuration, and cite the repository. We labelled your bot the
same way on our site.

If you already have results recorded against **v1.0**, they are
directly comparable: v1.1 took seven pre-registered directions at the errors
v1.0 left and shipped none of them, so the two releases play the identical
configuration tuple. The version names the release, not the policy. There is
nothing to re-run.
