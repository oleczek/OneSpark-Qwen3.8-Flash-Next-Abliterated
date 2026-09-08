#!/usr/bin/env bash
# Offline: no Docker, GPU, model downloads or HTTP requests.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for script in "$REPO"/*.sh; do bash -n "$script"; done
python3 -m unittest discover -s "$REPO/tests" -v
