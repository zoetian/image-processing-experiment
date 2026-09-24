#!/usr/bin/env bash
# Stylizes an image and writes the result to output/.
#
# URL defaults to the container started by `docker compose up`. When deployed
# through an OpenFaaS gateway use e.g.
#   URL=http://127.0.0.1:8080/function/process ./run.sh ...
set -euo pipefail

usage() {
    echo "usage: $0 image [style] [style_index]"
    echo "e.g.:  $0 process/raw_image/Rome.jpg varied 12"
    echo ""
    echo "style is monet (index 0-9) or varied (index 0-31)."
}

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    usage
    exit 1
fi

IMAGE=$1
STYLE=${2:-varied}
STYLE_INDEX=${3:-1}
URL=${URL:-http://127.0.0.1:8080}

mkdir -p output
OUT="output/${STYLE}_${STYLE_INDEX}_$(basename "$IMAGE")"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

if ! curl --fail-with-body -sS -X POST \
    -H "X-Style-Name: $STYLE" \
    -H "X-Style-Index: $STYLE_INDEX" \
    --data-binary "@$IMAGE" \
    -o "$TMP" \
    "$URL"; then
    cat "$TMP" >&2
    exit 1
fi

mv "$TMP" "$OUT"
echo "wrote $OUT"
