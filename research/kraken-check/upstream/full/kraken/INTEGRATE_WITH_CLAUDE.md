# Running KRAKEN on your own site — the paste-in brief

This file exists to be handed to a coding agent (Claude, Cursor, whatever you
use) so it can wire this bot into your project without you reading any of the
research code. Everything it needs is below; the full protocol reference is
`kraken/README.md` in the same directory.

Copy everything between the two rules into your agent, replacing the one
bracketed line with a sentence about your own project.

---

I want to add a second bot to my Fish/Literature project as an opponent. It is
called **KRAKEN v1.0** and it lives here:

    https://github.com/kv1514/fish-researchp12

Clone it somewhere outside my project, and **check out the branch
`claude/fishnbot-work-access-g7ciey`** — the default branch does not have the
`kraken/` directory yet:

    git clone https://github.com/kv1514/fish-researchp12
    cd fish-researchp12
    git checkout claude/fishnbot-work-access-g7ciey

The only three directories you need from it are `fish/`, `fish4/` and `kraken/` — 91 Python files, about
1.5 MB — and they must keep that layout relative to each other, because
`kraken/decide.py` puts the repository root on `sys.path` and imports the other
two. Requirements are Python 3.11+ and numpy. Nothing else: no network, no
model files, no GPU, no build step.

First, prove the thing works before writing any integration code:

    python -m kraken.decide --self-test

That takes about half a minute and must end with three lines reporting
"329 decisions compared, 0 mismatches", an off-turn line, and "ok". If it does
not, stop and show me the output — do not work around it.

Then read `kraken/README.md`. It documents the whole protocol. In short: the
bot is a **stateless decision service** speaking one JSON object per line on
stdin and one per line on stdout. My program stays the referee and the only
source of truth about the game; each request carries the entire public record
and the bot rebuilds its belief from scratch, so a dropped or duplicated
request cannot corrupt anything. A request looks like this:

    {"seat": 0,
     "hand": ["2C", "3C", "AH"],
     "hand_counts": [3, 9, 9, 9, 9, 9],
     "set_winner": [null, null, 0, 1, null, null, null, null, null],
     "turn": 0,
     "history": [ ...every public event so far, oldest first... ],
     "rules": {"wrong_distribution_outcome": "opponent"}}

and a reply looks like one of:

    {"action": {"type": "ask", "target": 3, "card": "2H"}}
    {"action": {"type": "declare", "half_suit": 3, "assignment": [0,0,2,4,4,4]}}
    {"action": {"type": "pass", "teammate": 2}}

Three things I specifically want you to get right, because each of them fails
silently rather than loudly:

1. **Cards are names, never integers.** "2C", "TD", "AH", "BJ"/"RJ" for the
   jokers. The bot rejects integers on purpose: my engine and this one number
   the deck over different permutations of the half-suits, so passing my own
   integer ids would have it play a scrambled hand — legally, badly, and with
   no error raised anywhere. It would look like a weak bot, not a broken
   integration. If you need the mapping, send `{"op": "cards"}` and it returns
   the full 54-name table plus the half-suit groupings; do not reimplement it.

2. **`history` must be complete and in chronological order**, oldest first,
   including asks that failed. The bot's whole strength is deduction from the
   public record, so a truncated or reordered history does not produce a worse
   move, it produces a move reasoned from a game that never happened. If the
   bot ever replies with an error about the event stream not being consistent
   with any deal, that is my log being wrong, not the bot being broken.

3. **`rules.wrong_distribution_outcome`** must match what my referee actually
   does with a misdeclared set. `"opponent"` means it goes to the other team
   (the standard rule); `"null"` means it is voided. Getting this wrong changes
   how willing the bot is to declare, which is where most of its edge is.

For determinism, pass a `"seed"` per request; the same seed and the same public
record always give the same move. Omit it and a seed is derived from the
history length, so a match is still reproducible from its own log.

Performance: keep one process alive for a whole match and write a line per
decision — startup is the slow part, a decision is not. It has no wall-clock
budget, no time control and no thinking-time parameter; the same position
always gets the same amount of work.

[Here describe my project: what language it is in, where the game loop lives,
and where you should add the opponent-selection code.]

---

## If you would rather host it the other way round

There is a second entry point, `fish4/decide.py`, which speaks a line-oriented
text protocol instead of JSON and is a deliberate mirror of the C++ shim this
project wrote to run dylann4500/fishbot v0.7 inside *our* arbiter. Use it if
you are driving from C++ and would rather not link a JSON library:

    printf 'RULES 9 opponent\nSEAT 0\nHAND 2S 3S 4S 5S 6S 7S 2H 3H 4H\nSEED 7\nDECIDE\n' \
        | python -m fish4.decide
    DECL 3 0 0 0 0 0 0

Same rules about names and about complete histories apply. The difference is
that this one *derives* hand counts and set winners from the event log rather
than accepting them, so the only thing a caller can send is the public record.
