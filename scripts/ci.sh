#!/usr/bin/env bash
# Local CI: byte-compile, lint (errors only), CPU tests. No GPU or mamba-ssm needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -m compileall -q src scripts tests
"$PY" -m ruff check --select E9,F63,F7,F82 src scripts tests
"$PY" -m pytest -q
