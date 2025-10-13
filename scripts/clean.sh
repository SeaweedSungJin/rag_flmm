#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Cleaning caches under: $ROOT_DIR"

find "$ROOT_DIR" \
  -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".ipynb_checkpoints" \) \
  -print -exec rm -rf {} +

find "$ROOT_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -print -delete

echo "Done."
