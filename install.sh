#!/usr/bin/env bash
# Adapted from azampatti/GB10-3.8-Flash-Next (Apache-2.0).
# Port hardening: pinned checkpoint, explicit downloads, generated overlays.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_linux
require_command python3
require_command docker
case "$MODE" in 1|2) ;; *) die 'MODE must be 1 (HashK) or 2 (PLE off)' ;; esac
docker info >/dev/null 2>&1 || die 'Cannot reach Docker daemon'

echo 'Verifying the pinned checkpoint (full SHA256 read, ~135 GB; no weight download)...'
MODEL_SNAPSHOT=$(python3 "$REPO/tools/checkpoint.py" snapshot --full)
export MODEL_SNAPSHOT
require_memory 105
if [ "$MODE" = 1 ] && [ ! -f "$REPO/ple_hashk_R4.pt" ]; then
  python3 -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; sys.exit(0 if free >= 40*2**30 else "Need 40 GiB free on the checkout volume for the artifact/build workspace")' "$REPO"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Pulling runtime image $IMAGE (multi-GB download)..."
  docker pull "$IMAGE"
fi
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
CAP=$(docker run --rm --gpus all --entrypoint python3 "$IMAGE_ID" -c \
  'import torch; print("%d%d" % torch.cuda.get_device_capability(0))')
[ "$CAP" = 121 ] || die "Expected GB10 / sm_121, found sm_$CAP. This is not a generic Blackwell recipe."
python3 "$REPO/tools/prepare_runtime.py" --image-id "$IMAGE_ID" --image-ref "$IMAGE"

if [ "$MODE" = 1 ]; then
  if [ -e "$REPO/ple_hashk_R4.pt" ] || [ -e "$REPO/ple_hashk_R4.pt.json" ]; then
    # Never quietly accept or overwrite a legacy/partial/different-model artifact.
    python3 "$REPO/tools/checkpoint.py" artifact
  else
    echo 'Building HashK R=4 on GPU. Stop other workloads first; the cgroup cap cannot guarantee host survival on UMA.'
    docker run --rm --gpus all --memory 90g --memory-swap 90g \
      -v "$HUB_CACHE:$HUB_CACHE:ro" -v "$MODEL_SNAPSHOT:$MODEL_SNAPSHOT:ro" \
      -v "$REPO:/repo:ro" -v "$REPO:/out" \
      -e "HASHK_MODEL_ID=$MODEL_ID" -e "HASHK_MODEL_REVISION=$MODEL_REVISION" \
      -e "HASHK_SNAPSHOT=$MODEL_SNAPSHOT" -e HASHK_NO_DOWNLOAD=1 \
      -e HF_HUB_OFFLINE=1 -e HASHK_OUT=/out/ple_hashk_R4.pt \
      --entrypoint python3 "$IMAGE_ID" /repo/tools/build_hashk_ple.py
    python3 "$REPO/tools/checkpoint.py" artifact
  fi
else
  echo 'MODE=2: skipped HashK build. PLE-off is a separate lossy configuration.'
fi
echo 'Setup complete. Set API_KEY, then run ./launch.sh (use the same MODE).'
