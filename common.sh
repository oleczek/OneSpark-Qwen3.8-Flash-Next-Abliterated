#!/usr/bin/env bash
# Local port configuration. No downloaded shell configuration is sourced.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4"
MODEL_REVISION="be794b990578ef3031eccf9f28e675a289a09ee9"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE/hub}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-$HUB_CACHE/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/$MODEL_REVISION}"
# linux/arm64 manifest of qwen38flashnext, resolved from Docker Hub 2026-09-07.
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:c93d57460ce7fca986c7c8150b5a2231b540ca226d17ef5563ac0dcee506c136}"
CONTAINER_NAME="${CONTAINER_NAME:-onespark-abliterated}"
OWNER_LABEL="org.onespark.abliterated.checkout"
PORT="${PORT:-30000}"
LISTEN_HOST="${LISTEN_HOST:-127.0.0.1}"
MODE="${MODE:-1}"
export MODEL_ID MODEL_REVISION MODEL_SNAPSHOT

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null || die "Missing command: $1"; }
require_key() {
  [ -n "${API_KEY:-}" ] || die 'Set API_KEY first (README quickstart). No shared default key.'
  [ "${#API_KEY}" -ge 16 ] || die 'API_KEY must have at least 16 characters.'
  case "$API_KEY" in *$'\n'*|*$'\r'*) die 'API_KEY must not contain newlines.' ;; esac
}
require_linux() { [ "$(uname -s)" = Linux ] || die 'Serving/building requires Linux on DGX Spark; use check.sh for offline checks.'; }
require_memory() {
  local ram swap
  ram=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  swap=$(awk '/^SwapFree:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${ram:-0}" -lt "$1" ]; then
    die "Only ${ram:-0} GiB RAM available; need at least $1 GiB for this conservative preflight. Stop other workloads. No automatic cleanup/reboot."
  fi
  if [ "${swap:-0}" -lt 16 ]; then
    printf 'WARNING: only %s GiB swap free; upstream recommends at least 16. Swap is not extra GPU capacity.\n' "${swap:-0}" >&2
  fi
}
container_owner() {
  docker inspect --format '{{ index .Config.Labels "org.onespark.abliterated.checkout" }}' "$CONTAINER_NAME"
}
