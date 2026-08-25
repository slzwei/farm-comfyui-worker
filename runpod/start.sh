#!/usr/bin/env bash
# Boot ComfyUI headless, wait until it answers, then hand over to the RunPod
# serverless handler. If ComfyUI dies during boot we exit non-zero so RunPod
# recycles the worker instead of accepting jobs that can only fail.
set -euo pipefail

COMFY_PORT="${COMFY_PORT:-8188}"

# Self-provision models onto the network volume if missing (no-op once
# populated; set SKIP_MODEL_CHECK=1 to bypass, e.g. baked-model images).
if [ "${SKIP_MODEL_CHECK:-0}" != "1" ]; then
  bash /app/scripts/ensure_models.sh
fi

echo "[start] launching ComfyUI on 127.0.0.1:${COMFY_PORT}"
python /app/ComfyUI/main.py \
  --listen 127.0.0.1 \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  ${COMFY_EXTRA_ARGS:-} &
COMFY_PID=$!

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    echo "[start] ComfyUI ready after ${i}s"
    break
  fi
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "[start] ComfyUI exited during boot" >&2
    exit 1
  fi
  if [ "$i" -eq 180 ]; then
    echo "[start] ComfyUI did not become ready within 180s" >&2
    exit 1
  fi
  sleep 1
done

echo "[start] starting RunPod serverless handler"
exec python -u /app/handler.py
