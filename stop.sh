#!/usr/bin/env bash
# Remove only the container started by this checkout; preserve all artifacts.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_command docker
docker info >/dev/null 2>&1 || die 'Cannot reach Docker daemon'
if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "No $CONTAINER_NAME container."
  exit 0
fi
[ "$(container_owner)" = "$REPO" ] || die "Refusing to stop $CONTAINER_NAME: not owned by this checkout ($REPO)"
docker stop --time 60 "$CONTAINER_NAME" >/dev/null
docker rm "$CONTAINER_NAME" >/dev/null
echo "Stopped and removed $CONTAINER_NAME. Checkpoint, HashK and logs on disk were not deleted."
