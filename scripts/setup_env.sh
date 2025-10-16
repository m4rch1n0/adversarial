#!/usr/bin/env bash
set -euo pipefail

# This script downloads the Imagenette 2-160 dataset into data/
# Usage: ./setup_env.sh

# Resolve project root as parent of this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ARCHIVE_NAME="imagenette2-160.tgz"
ARCHIVE_PATH="$DATA_DIR/$ARCHIVE_NAME"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

mkdir -p "$DATA_DIR"

if [[ -d "$DATA_DIR/imagenette2-160" ]]; then
  echo "Dataset already present at $DATA_DIR/imagenette2-160"
  exit 0
fi

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "Downloading Imagenette 2-160..."
  curl -L "$URL" -o "$ARCHIVE_PATH"
fi

echo "Extracting to $DATA_DIR ..."
tar -xzf "$ARCHIVE_PATH" -C "$DATA_DIR"

echo "Done. Dataset available at $DATA_DIR/imagenette2-160"

