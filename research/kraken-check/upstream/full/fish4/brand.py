"""The engine's name, in one place.

The name is user-facing and the identifiers are not, and the two must not be
confused.  ``V06_DEPLOYED`` in :mod:`fish4.registry4` and the ``"fishbot4"``
registry key are named in eight pre-registration documents, in every recorded
journal under ``results/``, and in forty job specs under ``jobs/``.  Those are
dated records: a pre-registration that names its arms before the data is worth
nothing if the arms can be renamed afterwards, so the identifiers are frozen
and this module carries the display name instead.

Renaming the bot is therefore a one-line edit here plus a run of the guard in
``tests4/test_brand.py``, which fails if the site, the handout or the paper
disagrees with these strings.
"""

#: The engine's name.  Set in caps because that is how it is set everywhere it
#: appears; ``NAME.title()`` is not a supported spelling.
NAME = "KRAKEN"

#: The version names the RELEASE -- this paper and the packages that ship
#: beside it -- and NOT the policy.  The distinction matters here because the
#: two have come apart.
#:
#: v1.0 was the configuration ``V06_DEPLOYED``: v0.6 plus the forced
#: exhaustive search that shipped on 2026-08-28.  The version moved with the
#: name because the thing being named was the first configuration that beat
#: Dylan's v0.7 by a margin measured over ten thousand games with a bridge
#: whose own contribution had been separately bounded.
#:
#: v1.1 then took seven pre-registered directions at the residue v1.0 left and
#: shipped NONE of them.  So the tuple this release plays is the same tuple
#: v1.0 played.  That is a result rather than an oversight, and it is recorded
#: here rather than left implicit, because the alternative is a reader seeing
#: "v1.1" on the site and on the handout and inferring that the policy moved.
VERSION_NUMBER = "1.1"
VERSION = f"v{VERSION_NUMBER}"

#: The last release at which the shipped configuration actually changed.
#: ``CONFIG_FINGERPRINT`` is what stops this from being a comment nobody
#: maintains: it pins the tuple, so a knob that finally clears its
#: pre-registered bar fails ``tests4/test_brand.py`` until whoever ships it
#: moves BOTH of these in the same commit.  Do not update the fingerprint on
#: its own -- a changed tuple with an unchanged CONFIG_UNCHANGED_SINCE is the
#: exact claim this pair exists to make false.
CONFIG_UNCHANGED_SINCE = "1.0"

#: sha256[:12] of ``V06_DEPLOYED`` rendered as
#: ``[key, dict(sorted(config.items()))]`` with compact separators.
CONFIG_FINGERPRINT = "3f46d74f9891"

FULL_NAME = f"{NAME} {VERSION}"

#: What it was called before 2026-08-28.  Kept so that a reader of an older
#: journal, an older commit or the pre-registration documents can tell that
#: they are looking at the same engine.
FORMER_NAME = "KV's FishBot v0.6"

#: The opponent, for symmetry: the only other named engine in this repository.
OPPONENT_NAME = "Dylan's FishBot v0.7"
