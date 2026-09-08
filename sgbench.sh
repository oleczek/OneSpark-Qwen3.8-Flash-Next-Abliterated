#!/usr/bin/env bash
# Prompt pools retained from azampatti/sgbench.sh (Apache-2.0).
# Emits JSON, exits nonzero on any failed stream. These are E2E rates.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$REPO/tools/api_probe.py" bench "$@"
