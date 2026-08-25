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
BASE_URL="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
LOCK_DIR="${VOLUME_ROOT}/.model-download-lock"
MARKER="${VOLUME_ROOT}/models/.provisioned"
STALE_SECONDS="${MODEL_LOCK_STALE_SECONDS:-180}"
WAIT_CAP_SECONDS="${MODEL_LOCK_WAIT_CAP_SECONDS:-1200}"

FILES=(
  "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "vae/wan2.2_vae.safetensors"
)

if [ ! -d "${VOLUME_ROOT}" ]; then
  echo "[models] FATAL: no network volume mounted at ${VOLUME_ROOT} — attach the model volume to the endpoint (runpod/README.md)" >&2
  exit 1
fi

all_present() {
  for rel in "${FILES[@]}"; do
    local path="${VOLUME_ROOT}/models/${rel}"
    [ -f "${path}" ] || return 1
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || echo 0)
    [ "${size}" -gt 1000000000 ] || return 1
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

for rel in "${FILES[@]}"; do
  dest="${VOLUME_ROOT}/models/${rel}"
  if [ -f "${dest}" ] && [ "$(stat -c%s "${dest}" 2>/dev/null || echo 0)" -gt 1000000000 ]; then
    echo "[models] have ${rel}"
    continue
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "[models] downloading ${rel}"
  curl -L -C - --fail --retry 5 --retry-delay 5 -o "${dest}" "${BASE_URL}/${rel}"
done

all_present || { echo "[models] download finished but verification failed" >&2; exit 1; }
touch "${MARKER}" 2>/dev/null || true
echo "[models] model set complete"
