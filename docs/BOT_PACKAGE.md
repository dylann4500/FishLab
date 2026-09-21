# Bringing your own bot — the FishLab bot package

This is the format for getting a bot you wrote onto a FishLab table. You zip it
up, somebody drops it on the setup screen, and it takes a seat: no fork of this
repository, no C++, no pull request, and no language requirement beyond "it can
read a line and write a line".

Everything here is implemented by `engine/src/botpkg.hpp` (the package),
`engine/src/extbot.hpp` (the protocol) and `engine/src/botcheck.hpp` (the
self-check). If a program disagrees with this document, the document is the
contract and the program is the bug — say so.

A complete working example is in [`examples/fishlab-bot-python`](../examples/fishlab-bot-python).
Copy it and replace the decision functions; the plumbing is the part that is
fiddly the first time.

---

## 1. The shape of a package

A package is a `.zip` containing a manifest called `fishbot.json` and whatever
else your bot needs — source, weights, data files. The manifest may sit at the
root of the zip or one directory down, so both of these work:

```
mybot.zip                       mybot.zip
├── fishbot.json                └── mybot/
├── bot.py                          ├── fishbot.json
└── weights.npz                     ├── bot.py
                                    └── weights.npz
```

The second is what you get from `zip -r mybot.zip mybot`, which is what most
people type, so it is supported deliberately rather than by accident. Files that
zip tools add on their own — `__MACOSX/`, `.DS_Store`, AppleDouble `._` stubs —
are dropped.

**Refused, every time:** absolute paths, `..` anywhere in a name, symbolic links,
encrypted entries, ZIP64, compression methods other than store and deflate.
A package is meant to be self-contained; a symlink is how an archive reaches out
of the directory it was told to unpack into.

**Limits:** 64 MB for the zip, 512 MB unpacked, 8192 files, 256 MB per file.

---

## 2. The manifest

```json
{
  "format": "fishlab-bot/1",
  "id": "sherlock",
  "name": "Sherlock",
  "version": "1.0",
  "author": "your name",
  "description": "One line about what it does.",
  "protocol": "fishlab-json-v1",
  "run": ["python3", "bot.py"],
  "python": { "venv": true, "requirements": "requirements.txt" },
  "env": { "OMP_NUM_THREADS": "1" },
  "timeout_ms": 10000,
  "poll_off_turn": true
}
```

| field | required | meaning |
|---|---|---|
| `format` | yes | Exactly `"fishlab-bot/1"`. |
| `name` | yes | What the table calls it. Up to 40 characters. |
| `run` | yes | argv, launched with the working directory set to your package. `run[0]` is either a bare program name looked up on `PATH` (`python3`, `node`) or a path inside the package (`bin/play`, `./play.sh`). It may not be absolute and may not contain `..`. A path inside the package is made executable on install, and any file the archive recorded as executable keeps that bit, so a shell wrapper or a compiled binary works. |
| `id` | no | The identifier used for `bot:<id>`, directory names and URLs. Lower case letters, digits and dashes, up to 32 characters. Derived from `name` when absent. |
| `version`, `author`, `description` | no | Shown on the setup screen. |
| `protocol` | no | `"fishlab-json-v1"` (default), or `"kv-json-v1"` — see §8. |
| `python.venv` | no | Ask the host to build a virtualenv for you. |
| `python.requirements` | no | The requirements file inside your package. Implies `venv`. |
| `env` | no | Environment variables for your process. Up to 32. |
| `timeout_ms` | no | How long the engine waits for one reply. Default 15000, clamped to 500–120000. Exceeding it is a fault, not a skipped move. |
| `poll_off_turn` | no | Default `true`. See §5.2 — set it to `false` only if you know what you are giving up. |

`PYTHONUNBUFFERED=1` and `FISHLAB_PROTOCOL` are set for you. Your process's
`stderr` is captured to a log you can read back, so print whatever you like to it.

---

## 3. The transport

Your program is started once and kept alive for the seat. It reads **one JSON
object per line** on stdin and writes **exactly one JSON object per line** on
stdout, in order, one reply per request.

The single most common first-attempt failure is buffering: a reply that sits in
your language's stdout buffer is indistinguishable from a bot that has hung, and
you will get a timeout fault that looks like a performance problem. Flush after
every reply. In Python that is `sys.stdout.flush()`; in Node, `process.stdout`
is already unbuffered to a pipe; in Go, flush your `bufio.Writer`.

Do not write anything else to stdout. Diagnostics go to stderr.

You may keep state between requests — you get your own process for the whole
seat — but you must never *need* to: every decision request carries the entire
public record, and a well-formed request must be answerable on its own.

---

## 4. The game, in one screen

Six seats, `0`–`5`. **Teams alternate**: seats 0, 2, 4 are one team, 1, 3, 5 the
other, so `team(seat) = seat & 1` and your teammates are `seat ± 2 (mod 6)`.

The deck is 54 cards in **nine half-suits of six**. Low and high of each suit,
plus the eights and the two jokers. Cards are always named, never numbered.

* **Ask.** On your turn you name a card and an opponent. It is legal only if you
  hold at least one card of that half-suit, do **not** hold the card asked for,
  and the target is an opponent who still has cards. A hit moves the card to you
  and you keep the turn; a miss passes the turn to the player you asked.
* **Declare.** At *any* moment — including during somebody else's turn — you may
  name a half-suit and say which of **your own three seats** holds each of its
  six cards. Right: your team scores it. Wrong: the other team does. Either way
  the half-suit leaves play. You need hold none of it to declare it.
* **Out of cards.** A player with no cards can no longer ask or be asked, but may
  still declare. If it is such a player's turn, they choose which teammate who
  still has cards receives it.
* **The forced endgame.** When one whole team is out of cards, the other team
  must declare every remaining half-suit with no further asking. The engine
  sweeps a ladder of confidence thresholds, asking each seat whether it is
  willing to declare a given half-suit at that threshold, and finally demands an
  answer from somebody. Only the willingness bit crosses between teammates.

The complete rules as implemented are `engine/src/fish.hpp` and
`engine/src/game.hpp`; [`docs/PLAY.md`](PLAY.md) is the human-facing version.

### Card names

Rank then suit, `S H D C`. Ten is **`T`**, not `10`.

| half-suit | index | cards |
|---|---|---|
| Low Spades | 0 | `2S 3S 4S 5S 6S 7S` |
| High Spades | 1 | `9S TS JS QS KS AS` |
| Low Hearts | 2 | `2H 3H 4H 5H 6H 7H` |
| High Hearts | 3 | `9H TH JH QH KH AH` |
| Low Diamonds | 4 | `2D 3D 4D 5D 6D 7D` |
| High Diamonds | 5 | `9D TD JD QD KD AD` |
| Low Clubs | 6 | `2C 3C 4C 5C 6C 7C` |
| High Clubs | 7 | `9C TC JC QC KC AC` |
| Eights & Jokers | 8 | `8S 8H 8D 8C RJ BJ` |

You do not have to hardcode this. The handshake hands you the whole deck in
engine order, and the rule that relates the two is:

> the card at index `i` of the `cards` array belongs to half-suit `i / 6`, at
> position `i % 6` within it — and `owner[j]` in a declaration is the seat
> holding `cards[set * 6 + j]`.

---

## 5. The protocol — `fishlab-json-v1`

### 5.1 `hello`

Sent once, before anything else.

```json
{"op":"hello","protocol":"fishlab-json-v1","engine":"fishlab","seats":6,"set_size":6,
 "timeout_ms":10000,"cards":["2S","3S", "... 54 names ..."],"sets":["Low Spades", "..."]}
```

Reply:

```json
{"ok":true,"name":"Sherlock","version":"1.0","protocol":"fishlab-json-v1"}
```

If you send a `protocol` that is not `fishlab-json-v1`, the engine stops and says
so. Omitting it is taken as agreement.

### 5.2 The four decision requests

Each carries a `state` object (§6). Answer with exactly one action.

**`ask`** — it is your turn and you must ask.

```json
{"op":"ask","state":{ … }}
→ {"action":"ask","card":"QH","target":3}
```

**`declare_poll`** — *may* you want to declare right now? Sent to **every seat
before every move**, because the rules let anybody declare at any moment — and
again after each declaration that lands, until nobody wants another.

```json
{"op":"declare_poll","state":{ … }}
→ {"action":"none"}
→ {"action":"declare","set":4,"owner":[0,2,0,4,2,0],"confidence":1.0}
```

This is the request that arrives most often — five times out of six it is not
even your turn — so make the "no" path cheap. If you would rather not be asked
at all, set `"poll_off_turn": false` in the manifest and you will only be polled
on your own turn. That is a real strategic cost: a declaration you can prove
during an opponent's turn is one you would otherwise have to sit on.

**`pass`** — you have no cards and hold the turn; give it to a teammate. The
legal choices are listed, and you must name one of them.

```json
{"op":"pass","candidates":[2,4],"state":{ … }}
→ {"action":"pass","to":2}
```

**`forced`** — the forced endgame. `set` is the half-suit being asked about,
`threshold` is the confidence the engine is sweeping at, and `last_resort` says
that somebody on your team must answer now.

```json
{"op":"forced","set":3,"threshold":0.9,"last_resort":false,"state":{ … }}
→ {"action":"none"}
→ {"action":"declare","set":3,"owner":[1,3,1,1,5,3],"confidence":0.72}
```

Rules for this one:

* You may only answer about the `set` you were asked about. A declaration naming
  a different half-suit is declined and the sweep comes back round for that one.
* The engine compares your `confidence` against `threshold` itself. Reporting a
  number you believe is more useful than clamping it to 1.
* **When `last_resort` is true, answer.** If you decline, the engine has to
  allocate on your behalf, and its fallback names every card to one seat, which
  is nearly always wrong — and it will be recorded as *your* declaration. The
  self-check counts these and complains.

### 5.3 `new_game`

Sent once per deal, before the first decision, so a bot that caches per game
knows to drop it.

```json
{"op":"new_game","seat":2,"deck_sets":9,"hand":["2S","QH", "..."],"rules":{ … }}
→ {"ok":true}
```

### 5.4 Saying no

Any reply may be `{"error":"..."}`. The engine stops the game and shows your
message, which is the right thing to send when you have genuinely no legal move
— far better than inventing one.

---

## 6. The `state` object

```json
{
  "seat": 2,
  "turn": 2,
  "deck_sets": 9,
  "hand": ["2S","QH","8D"],
  "hand_counts": [9,9,8,10,9,9],
  "score": [1,0],
  "set_active": [false,true,true,true,true,true,true,true,true],
  "set_winner": [0,null,null,null,null,null,null,null,null],
  "n_asks": 12,
  "rules": {"out_of_turn_declare":true,"cardless_may_declare":true,
            "max_asks":400,"deck_sets":9},
  "history": [ … ]
}
```

`turn` is whose move it is. In a `forced` request it is set to your own seat,
because the engine has genuinely handed you the move without moving the table's
turn marker. `set_winner[s]` is the team that took half-suit `s`, or `null` while
it is still in play.

### Events in `history`

```json
{"t":"ask","actor":0,"target":1,"card":"9D","success":false,"counts":[9,9,9,9,9,9]}
{"t":"declare","actor":3,"set":4,"forced":false,"success":true,"winner":1,
 "owner":[1,3,1,5,3,1],"counts":[9,9,9,6,9,7]}
{"t":"pass","actor":2,"target":4,"counts":[0,9,9,7,9,8]}
```

`counts` is every seat's hand size **after** the event.

On a declaration, `owner` is what the declarer **claimed**, which everyone at the
table hears, and `success` says whether it was right. **A wrong declaration
reveals nothing else** — not who really held the cards. That is what a person
sitting at the table sees, and it is all your bot gets. If your model needs the
true holders at resolution, it is modelling a different game; see §8.

There is no event for a half-suit leaving play beyond the declaration itself, and
none for the deal.

---

## 7. What happens when your bot is wrong

The engine **never** replaces a bad answer with a legal move. A host that
substitutes quietly gives you a bot that loses every game for reasons you cannot
see, which is worse than no bot at all. So every one of these stops the game and
reports the exact request and the exact reply:

* a reply that is not one line of JSON, or not an object;
* a reply that does not answer the question asked (an `ask` request answered with
  a declaration);
* an ask that is illegal, with the specific rule named — *you already hold 6H*,
  *seat 1 is on your own team*, *you hold no card of half-suit 4*;
* a declaration naming a half-suit out of play, a seat out of range, or **a seat
  on the other team** — the most common indexing mistake, and the one worth
  catching loudly: the engine's own driver simply *skips* an allocation that
  names the wrong team, so a bot with its seats transposed would look like a bot
  that has decided never to declare, which is a much harder thing to debug than
  an error message;
* a `pass` that does not name one of the offered candidates;
* no reply within `timeout_ms`;
* the process dying.

At an interactive table that ends **that game**, with the message on the felt.
In a batch run (`fish match`) it ends the run, because a measured number must
never come from a game in which a move was substituted.

---

## 8. If your bot already speaks KV's dialect

Set `"protocol": "kv-json-v1"` and `"run": ["python3","-m","yourpackage.decide"]`.
The package is then played through `engine/src/kv6.hpp`, the bridge written for
KV's `fishbot_v06`/`kraken` services, and nothing on your side has to change.

Two differences you inherit, both documented at the top of that file: the two
projects number their half-suits differently (the bridge derives the whole
correspondence from your own `{"op":"cards"}` table at startup and refuses to run
if the decks disagree), and their `ClaimEvent` wants the true holders at
resolution, which this engine does not publish on a wrong declaration. New bots
should use `fishlab-json-v1`, where that question does not arise.

Two things that do apply to an uploaded KV-dialect package and not to the host's
own vendored one: your `timeout_ms` is enforced on the first read of every reply,
and your `stderr` is captured to the same log the panel shows. `run` must be
`["<python>", "-m", "<module>"]` — that bridge launches an interpreter and a
module rather than an arbitrary argv, which is the shape KV's own packages ship.

---

## 9. Getting it onto a table

### From the browser

`fish serve`, then **Bots** on the setup screen: drop the zip, and it appears in
every seat's engine list.

At a shared table (`--lan` / `--public`) anyone with the invite code may upload,
because uploading runs nothing. Only the host can check a bot, install its
dependencies, seat it or remove it — running somebody's code is a decision the
person whose laptop it is gets to make. `--lock-bots` narrows uploading to the
host too.

### From the command line

```bash
cd engine
./fish bots add ../examples/fishlab-bot-python.zip
./fish bots list
./fish bots prepare sherlock     # only if the manifest asks for a virtualenv
./fish bots check sherlock
```

`check` plays complete games against a cheap baseline and tells you what it
actually exercised:

```
Played 4 complete games in 0.06 s.
  seat 0 (Sherlock) against five hunter
  first deal: half-suits 4 - 5 (its team first), 72 asks over 81 events
  answered: 55 ask, 333 declaration poll, 0 pass, 15 forced
  it declared 2 half-suits of its own accord
  0.1 ms per reply on average
```

It names what it could *not* reach, too, so a clean report is not mistaken for
coverage of a branch that never ran.

### And then measure it

An installed package is a policy spec like any other, so the whole measurement
apparatus is already pointed at it:

```bash
./fish match --a=bot:sherlock --b=v06 --games=400 --rotations=6
```

Same duplicate-block design, same confidence intervals, same everything the
papers were produced with.

---

## 9b. Packaging one of *our* policies

The traffic goes both ways. `fish botserve` speaks this same protocol on
stdin/stdout and drives any policy the engine can build, so SESTINA v1.0 (or
v0.6, or a baseline, or an ablation) can be zipped into an ordinary
`fishlab-bot/1` package and seated through the upload path like anybody else's
entry — or handed to somebody else's table to practise against.

```bash
scripts/package-bot.sh                       # SESTINA v1.0, the frozen v0.7
scripts/package-bot.sh v06 fishbot-v06 "FishBot v0.6" 0.6
```

The default reads its spec straight out of `engine/fishbot_v07.json`, so the
package is the frozen configuration by construction rather than a spec string
copied by hand. The zip contains the manifest, the engine binary and a README.

Two properties worth stating, because they are the ones a competitor will ask
about:

* **It is not a reimplementation.** `botserve` calls the same `makeAgent()` the
  table calls, so what plays is the policy itself.
* **It is not bit-identical to a native seat, and cannot be.** A seat in-process
  is seeded from the deal (`mixSeed(seed, p + 77)`); this protocol deliberately
  does not carry the deal seed, because a bot that could read it could invert it
  and recover the hands. A searching policy therefore explores different
  determinizations over the wire than it does natively. Measured over 400 games
  against v0.6 on the same seeds, the packaged and native forms agree to within
  well under the paired noise floor — 54.0% [49.1, 58.8] against 54.8% [49.9,
  59.6] — and head to head against its own native self the package is a coin
  flip at 53.0% [48.1, 57.8].

The binary is native to whatever built it, so a package built on an arm64 Mac
will not run on an x86 Linux box; rebuild there and re-run the script.

---

## 10. Dependencies

If your bot needs packages, ship `requirements.txt` and set
`"python": {"venv": true}`. The host clicks **Install deps** (or runs
`fish bots prepare <id>`) and the engine builds a virtualenv *inside your
package* and installs into it, then substitutes that interpreter for a leading
`python`/`python3` in your `run`. The full pip transcript is readable from the
same panel, because "pip failed" without the output is a support ticket rather
than an answer.

This step runs code — `pip install` executes what it downloads — which is why it
is the host's to trigger and never automatic.

The alternative is to have no dependencies at all. The reference bot is pure
standard library on purpose.

---

## 11. What this costs the person hosting

Worth stating plainly, because it is the honest description of the feature:

* **Installing a package executes nothing.** Unzipping is unzipping. Your code
  sits inert on disk.
* **Seating a bot runs your program on the host's machine**, with the host's
  permissions, in your package's directory. There is no sandbox. It is the same
  trust as `git clone && python bot.py`, which is what everybody was doing by
  hand before this existed — the difference is that it now takes a drag and a
  drop, so the boundary is worth saying out loud rather than leaving implied.
* The host decides what to run. The invite code buys a seat at a card game, not
  execution.
