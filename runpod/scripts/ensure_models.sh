#!/usr/bin/env bash
# Self-provision the Wan 2.2 TI2V-5B model set onto the attached network
# volume if it's missing. Runs on worker boot (start.sh) so a fresh volume
# needs no manual pod session — the first cold start downloads ~18GB once,
# and every later boot finds the files and skips straight through.
#
# Locking (hardened after a real incident): serverless workers can be
# SIGKILLed by host reclaims mid-download, which would leave a zombie lock.
# The downloader therefore refreshes the lock's mtime from a background
# heartbeat every 20s; a lock without a heartbeat for STALE_SECONDS is dead
# and gets stolen. Waiters also give up waiting after WAIT_CAP_SECONDS and
# steal — curl -C - resumes partial files, so steals are always safe.
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
BASE_URL="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
TURBO_URL="https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main"
LOCK_DIR="${VOLUME_ROOT}/.model-download-lock"
# Marker is version-stamped: switching model sets invalidates it automatically.
MARKER="${VOLUME_ROOT}/models/.provisioned-minimax-h3-v1"
STALE_SECONDS="${MODEL_LOCK_STALE_SECONDS:-180}"
WAIT_CAP_SECONDS="${MODEL_LOCK_WAIT_CAP_SECONDS:-1200}"

# MiniMax H3 FL2VA (image-to-video) set — ~44.4GB total. Sizes are the exact
# byte counts from the HuggingFace tree API; a short file means a truncated
# download and is re-fetched.
FILES=(
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|20971520000"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|15690000000"
  "vae/minimax_h3_video_vae_fp16.safetensors|5210000000"
  "vae/minimax_h3_audio_vae_fp32.safetensors|610000000"
  "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors|1960000000"
)

# The turbo LoRA lives in a different repo than the base model files.
url_for() {
  case "$1" in
    loras/*) echo "${TURBO_URL}/$(basename "$1")" ;;
    *) echo "${BASE_URL}/$1" ;;
  esac
}

if [ ! -d "${VOLUME_ROOT}" ]; then
  echo "[models] FATAL: no network volume mounted at ${VOLUME_ROOT} — attach the model volume to the endpoint (runpod/README.md)" >&2
  exit 1
fi

all_present() {
  for entry in "${FILES[@]}"; do
    local rel="${entry%%|*}" min="${entry##*|}"
    local path="${VOLUME_ROOT}/models/${rel}"
    [ -f "${path}" ] || return 1
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || echo 0)
    # Per-file floor: a truncated multi-GB checkpoint loads as noise or
    # crashes ComfyUI, so each file is checked against its own real size.
    [ "${size}" -ge "${min}" ] || return 1
  done
  return 0
}

if [ -f "${MARKER}" ] && all_present; then
  echo "[models] provisioned marker present — all model files on ${VOLUME_ROOT}"
  exit 0
fi
if all_present; then
  echo "[models] all model files present on ${VOLUME_ROOT}"
  touch "${MARKER}" 2>/dev/null || true
  exit 0
fi

lock_age() {
  echo $(( $(date +%s) - $(stat -c %Y "${LOCK_DIR}" 2>/dev/null || date +%s) ))
}

echo "[models] models missing — acquiring download lock"
waited=0
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  age=$(lock_age)
  if [ "${age}" -gt "${STALE_SECONDS}" ]; then
    echo "[models] lock heartbeat dead (age ${age}s > ${STALE_SECONDS}s) — stealing"
    rm -rf "${LOCK_DIR}" || true
    continue
  fi
  if [ "${waited}" -ge "${WAIT_CAP_SECONDS}" ]; then
    echo "[models] waited ${waited}s — stealing lock regardless (resume is safe)"
    rm -rf "${LOCK_DIR}" || true
    continue
  fi
  echo "[models] another worker is downloading (heartbeat ${age}s ago) — waiting"
  sleep 15
  waited=$(( waited + 15 ))
  if all_present; then
    echo "[models] models appeared while waiting"
    touch "${MARKER}" 2>/dev/null || true
    exit 0
  fi
done

# Heartbeat: prove this downloader is alive; dies with the script.
( while :; do touch "${LOCK_DIR}" 2>/dev/null || exit 0; sleep 20; done ) &
HEARTBEAT_PID=$!
trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true; rm -rf "${LOCK_DIR}"' EXIT

for entry in "${FILES[@]}"; do
  rel="${entry%%|*}"
  min="${entry##*|}"
  dest="${VOLUME_ROOT}/models/${rel}"
  if [ -f "${dest}" ] && [ "$(stat -c%s "${dest}" 2>/dev/null || echo 0)" -ge "${min}" ]; then
    echo "[models] have ${rel}"
    continue
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "[models] downloading ${rel} (~$((min / 1000000000))GB)"
  # aria2 (8 parallel segments) when available — a single stream from
  # HuggingFace is throttled and 44GB takes hours; curl -C - is the fallback.
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x8 -s8 -k4M --file-allocation=none --console-log-level=warn -c \
      -d "$(dirname "${dest}")" -o "$(basename "${dest}")" "$(url_for "${rel}")" \
      || curl -L -C - --fail --retry 5 --retry-delay 5 -o "${dest}" "$(url_for "${rel}")"
  else
    curl -L -C - --fail --retry 5 --retry-delay 5 -o "${dest}" "$(url_for "${rel}")"
  fi
done

all_present || { echo "[models] download finished but verification failed" >&2; exit 1; }
touch "${MARKER}" 2>/dev/null || true
echo "[models] model set complete"
