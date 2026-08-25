#!/usr/bin/env bash
# Self-provision the Wan 2.2 TI2V-5B model set onto the attached network
# volume if it's missing. Runs on worker boot (start.sh) so a fresh volume
# needs no manual pod session — the first cold start downloads ~18GB once
# (datacenter → HuggingFace is fast; expect ~5-15 min, billed once), and
# every later boot finds the files and skips straight through.
#
# A mkdir-based lock keeps concurrent cold workers from downloading twice;
# waiters poll until the winner finishes. Locks older than 90 min are
# considered stale (crashed downloader) and are stolen.
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
BASE_URL="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
LOCK_DIR="${VOLUME_ROOT}/.model-download-lock"
STALE_SECONDS=5400

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
    [ "${size}" -gt 1000000 ] || return 1
  done
  return 0
}

if all_present; then
  echo "[models] all model files present on ${VOLUME_ROOT}"
  exit 0
fi

echo "[models] models missing — acquiring download lock"
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  lock_age=$(( $(date +%s) - $(stat -c %Y "${LOCK_DIR}" 2>/dev/null || date +%s) ))
  if [ "${lock_age}" -gt "${STALE_SECONDS}" ]; then
    echo "[models] stealing stale lock (age ${lock_age}s)"
    rm -rf "${LOCK_DIR}" || true
    continue
  fi
  echo "[models] another worker is downloading (lock age ${lock_age}s) — waiting"
  sleep 15
  if all_present; then
    echo "[models] models appeared while waiting"
    exit 0
  fi
done
trap 'rm -rf "${LOCK_DIR}"' EXIT

for rel in "${FILES[@]}"; do
  dest="${VOLUME_ROOT}/models/${rel}"
  if [ -f "${dest}" ] && [ "$(stat -c%s "${dest}" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "[models] have ${rel}"
    continue
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "[models] downloading ${rel}"
  curl -L -C - --fail --retry 5 --retry-delay 5 -o "${dest}" "${BASE_URL}/${rel}"
done

all_present || { echo "[models] download finished but verification failed" >&2; exit 1; }
echo "[models] model set complete"
