#!/bin/sh
set -eu

DEMO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE_NAME=${HARMPROFILE_IMAGE:-harmprofile-demo}

if [ "$#" -lt 1 ]; then
    echo 'Usage: ./demo/generate.sh "high-level description" [options]' >&2
    exit 2
fi
if [ ! -f "$DEMO_DIR/.env" ]; then
    echo "Missing $DEMO_DIR/.env; copy .env.example and set OPENROUTER_API_KEY." >&2
    exit 2
fi

mkdir -p "$DEMO_DIR/runs"
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$DEMO_DIR"
echo "Starting HarmProfile generation agent"
exec docker run --rm -it \
    --env-file "$DEMO_DIR/.env" \
    -v "$DEMO_DIR/runs:/runs" \
    "$IMAGE_NAME" generate "$@"
