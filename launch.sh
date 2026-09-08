#!/usr/bin/env bash
# SGLang/HashK/FP8 flag stack adapted from azampatti (Apache-2.0).
# No model downloads and no removal of existing containers.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_linux
require_command python3
require_command docker
require_command curl
require_key
MEM_FRACTION="${MEM_FRACTION:-0.90}"
THINKING="${THINKING:-medium}"
case "$THINKING" in off|low|medium|xhigh) ;; *) die 'THINKING must be off, low, medium or xhigh' ;; esac
case "$MODE" in
  1) CTX="${CTX:-200000}"
     python3 "$REPO/tools/checkpoint.py" artifact
     PLE_ENV=(-e SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt) ;;
  2) CTX="${CTX:-256000}"
     PLE_ENV=(-e SGLANG_QWEN4_PLE_OFF=1) ;;
  *) die 'MODE must be 1 (HashK) or 2 (PLE off)' ;;
esac
python3 -c 'import sys; p,c,m=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]); sys.exit(0 if 1 <= p <= 65535 and 1 <= c <= 262144 and 0 < m <= 0.90 else "Invalid PORT/CTX/MEM_FRACTION (max context 262144, max fraction 0.90)")' "$PORT" "$CTX" "$MEM_FRACTION"
MODEL_SNAPSHOT=$(python3 "$REPO/tools/checkpoint.py" snapshot)
IMAGE_ID=$(python3 "$REPO/tools/prepare_runtime.py" --check)
docker info >/dev/null 2>&1 || die 'Cannot reach Docker daemon'
docker image inspect "$IMAGE_ID" >/dev/null 2>&1 || die 'Installed image was removed; run ./install.sh'
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  die "Container $CONTAINER_NAME already exists. Inspect it; ./stop.sh only removes containers owned by this checkout."
fi
require_memory 105
# Fail early if another service already owns the listening address/port.
python3 -c 'import socket,sys; s=socket.socket(); s.bind((sys.argv[1],int(sys.argv[2]))); s.close()' "$LISTEN_HOST" "$PORT"
if [ "$THINKING" = off ]; then
  KW='{"enable_thinking":false}'
else
  KW="{\"enable_thinking\":true,\"reasoning_effort\":\"$THINKING\"}"
fi
docker run -d --name "$CONTAINER_NAME" --label "$OWNER_LABEL=$REPO" \
  --gpus all --network host --ipc=host --shm-size 32g \
  -v "$HUB_CACHE:$HUB_CACHE:ro" -v "$MODEL_SNAPSHOT:$MODEL_SNAPSHOT:ro" \
  -v "$REPO:/patches:ro" \
  -v "$REPO/.state/qwen4_exp.py:/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py:ro" \
  -v "$REPO/patches/flash_fwd.py:/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py:ro" \
  -v "$REPO/patches/qwen_sparse_attn_backend.py:/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro" \
  -v "$REPO/patches/sparse_attn.py:/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py:ro" \
  "${PLE_ENV[@]}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e Q4X_FP8=lm_head,linear_attn,qkv_proj,o_proj,shared_expert \
  "$IMAGE_ID" python3 -m sglang.launch_server \
    --model-path "$MODEL_SNAPSHOT" --served-model-name "$MODEL_ID" --trust-remote-code --language-only \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
    --chunked-prefill-size 8192 --max-prefill-tokens 32768 \
    --max-running-requests 8 --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 \
    --context-length "$CTX" --mem-fraction-static "$MEM_FRACTION" \
    --default-chat-template-kwargs "$KW" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --strip-thinking-cache \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --api-key "$API_KEY" --host "$LISTEN_HOST" --port "$PORT"
python3 "$REPO/tools/prepare_runtime.py" --record-launch \
  "$MODE" "$CTX" "$MEM_FRACTION" "$THINKING" "$PORT" "$LISTEN_HOST" "$CONTAINER_NAME"

HEALTH_HOST="$LISTEN_HOST"
[ "$HEALTH_HOST" != 0.0.0.0 ] || HEALTH_HOST=127.0.0.1
echo "Booting; watch: docker logs -f $CONTAINER_NAME"
for ((i=0; i<120; i++)); do
  [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" = true ] \
    || die "Container exited; inspect docker logs $CONTAINER_NAME"
  if curl --fail --silent --max-time 5 -H "Authorization: Bearer $API_KEY" \
    "http://$HEALTH_HOST:$PORT/health" >/dev/null; then
    echo "Health OK on $LISTEN_HOST:$PORT. Now run ./smoke.sh; health alone does not prove generation works."
    exit 0
  fi
  sleep 15
done
die "Not healthy within the polling window; inspect docker logs $CONTAINER_NAME (container left for diagnosis)"
