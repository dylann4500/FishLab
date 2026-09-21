# Dylan's FishBot v0.7, bridged

The exhibition opponent on the website: **FishBot v0.7** from
<https://github.com/dylann4500/fishbot>, running its own C++ engine code,
playing our game through a one-shot decision binary.

- `UPSTREAM.txt` — the upstream commit this build is pinned to.
- `v07_spec.txt` — their frozen `allparamsSpec` (from `engine/fishbot_v07.json`),
  the exact parameter vector plus det=12 test-time search their repo names as
  v0.7. Also embedded in `fish4/dylan_v07.py` for the deployed function.
- `shim_decide.cpp` — the shim `main`: feed it seat, hand, rules and the public
  event stream; it answers one decision (ask, declaration, or pass choice)
  through their `factory.hpp` agent and exits. Compiles against their headers
  only; none of their networking/server code (`httpd`, `serve`, `tunnel`,
  `lobby`) is included in the build.
- `build.sh` — local build. The deployed copy at `api/bin/fish_v07_decide` is
  built with `-static` and no `-march=native` so it runs on the Vercel runtime.

The Python side is `fish4/dylan_v07.py`, registered as `dylan_v07`. The two
projects now agree on misdeclaration scoring — a wrong declaration awards the
set to the opponents in both engines (this repo's baseline since the rule
correction; theirs always) — so the one remaining rule difference is their
out-of-turn declaration channel versus our declarations-on-turn, documented
there. Our engine arbitrates, their policy is told the rules it is actually
playing under, and every proposed action is legality-checked with a counted
fallback.
Verification: full games with zero fallbacks across 20 mixed 3v3 games before
anything shipped.

Their repository carries no licence file; the code is used here to run their
bot as-published for a head-to-head exhibition, with attribution, and none of
it is redistributed in this repository beyond the compiled decide binary.
