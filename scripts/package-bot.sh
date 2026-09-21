#!/bin/sh
# Package one of this engine's own policies as an uploadable bot.
#
# The result is an ordinary fishlab-bot/1 package (docs/BOT_PACKAGE.md): a
# manifest, this engine's binary, and a README. It runs `fish botserve`, which
# speaks the same protocol a foreign bot speaks, so the packaged policy is
# seated through the ordinary upload path rather than through a special case --
# and can be handed to somebody else's table.
#
#   scripts/package-bot.sh                       # SESTINA v1.0, the frozen v0.7
#   scripts/package-bot.sh v06 fishbot-v06 "FishBot v0.6" 0.6
#
# The default reads its spec from engine/fishbot_v07.json, so the package is the
# frozen configuration by construction and not a spec string copied by hand.
set -eu

cd "$(dirname "$0")/.."
ENGINE=engine
OUT=${OUT:-dist/bots}

SPEC=${1:-}
ID=${2:-sestina}
NAME=${3:-SESTINA v1.0}
VERSION=${4:-1.0}

if [ -z "$SPEC" ]; then
  SPEC=$(python3 -c "import json;print(json.load(open('$ENGINE/fishbot_v07.json'))['spec'])")
fi

if [ ! -x "$ENGINE/fish" ]; then
  echo "package-bot: $ENGINE/fish is not built -- run (cd engine && make)" >&2
  exit 1
fi

ARCH=$(uname -s)-$(uname -m)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$ID" "$OUT"

cp "$ENGINE/fish" "$STAGE/$ID/fish"
chmod +x "$STAGE/$ID/fish"

SPEC="$SPEC" ID="$ID" NAME="$NAME" VERSION="$VERSION" STAGE="$STAGE" python3 - <<'PY'
import json, os
stage, bid = os.environ['STAGE'], os.environ['ID']
name, version, spec = os.environ['NAME'], os.environ['VERSION'], os.environ['SPEC']
manifest = {
    "format": "fishlab-bot/1",
    "id": bid,
    "name": name,
    "version": version,
    "author": "FishLab",
    "description": "The FishLab engine's own policy, served over the bot protocol. "
                   "Spec: " + spec,
    "protocol": "fishlab-json-v1",
    "run": ["./fish", "botserve", "--policy=" + spec,
            "--name=" + name, "--botversion=" + version],
    "timeout_ms": 20000,
}
with open(os.path.join(stage, bid, "fishbot.json"), "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

cat > "$STAGE/$ID/README.md" <<EOF
# $NAME — as a FishLab bot package

This is the FishLab engine itself, packaged in the \`fishlab-bot/1\` format
(see \`docs/BOT_PACKAGE.md\`) so it can be uploaded to any FishLab table and
seated like anybody else's entry.

    spec         $SPEC
    protocol     fishlab-json-v1
    built for    $ARCH

\`fish botserve\` speaks the bot protocol on stdin/stdout and drives the policy
through the engine's own \`makeAgent\`, so what plays here is the policy and not
a reimplementation of it.

**It is not bit-identical to a native seat, and cannot be.** A seat in-process is
seeded from the deal; the protocol deliberately withholds the deal seed, because
a bot that could read it could invert it and recover the hands. A searching
policy therefore explores different determinizations here than it would natively
— the same policy at the same strength, over different rollouts. The seed used
is fixed, so a run of this package is reproducible on its own terms.

**\`fish\` is a native $ARCH binary.** On a different platform, build the engine
from source (\`cd engine && make\`) and re-run \`scripts/package-bot.sh\`.
EOF

ZIP="$OUT/$ID.zip"
rm -f "$ZIP"
( cd "$STAGE" && zip -qr "$OLDPWD/$ZIP" "$ID" )

echo "$ZIP"
ls -lh "$ZIP" | awk '{print "  " $5}'
echo "  id      $ID"
echo "  name    $NAME $VERSION"
echo "  spec    $SPEC"
echo "  arch    $ARCH"
echo
echo "  install it:  (cd engine && ./fish bots add ../$ZIP && ./fish bots check $ID)"
