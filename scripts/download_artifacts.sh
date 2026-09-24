#!/usr/bin/env bash
# Downloads the prebuilt vector index (book.db) from the GitHub Release.
# The fine-tuned embedder is pulled automatically from Hugging Face Hub
# (lau299/ximhai-embedder-es) the first time rag_cli.py runs.
set -euo pipefail

REPO="lauro299/hackaton-devf-llama"
TAG="v1.0-hackathon"
DEST="${1:-book.db}"

if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern book.db --output "$DEST" --clobber
else
    curl -fL -o "$DEST" "https://github.com/$REPO/releases/download/$TAG/book.db"
fi

echo "Saved $DEST"
