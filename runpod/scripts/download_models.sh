#!/usr/bin/env bash
# Populate the RunPod network volume with the Wan 2.2 TI2V-5B model set
# (~18GB total). Run ONCE from any pod that has the volume attached (a cheap
# CPU pod works):
#
#   bash download_models.sh [/runpod-volume]
#
# Files come from Comfy-Org's repackaged release — the exact names the
# product-demo-v1 workflow references. curl -C - resumes partial downloads.
set -euo pipefail

VOLUME_ROOT="${1:-/runpod-volume}"
BASE_URL="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

declare -a FILES=(
  "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "vae/wan2.2_vae.safetensors"
)

for rel in "${FILES[@]}"; do
  dest="${VOLUME_ROOT}/models/${rel}"
  mkdir -p "$(dirname "${dest}")"
  echo "==> ${rel}"
  curl -L -C - --fail --retry 5 --retry-delay 5 -o "${dest}" "${BASE_URL}/${rel}"
done

echo
echo "Done. Volume contents:"
find "${VOLUME_ROOT}/models" -type f -exec du -h {} \;
echo
echo "Sanity check — every file must be non-trivially sized (the 5B model is ~10GB):"
for rel in "${FILES[@]}"; do
  size=$(stat -c%s "${VOLUME_ROOT}/models/${rel}" 2>/dev/null || stat -f%z "${VOLUME_ROOT}/models/${rel}")
  if [ "${size}" -lt 1000000 ]; then
    echo "WARNING: ${rel} is only ${size} bytes — download likely failed" >&2
    exit 1
  fi
done
echo "All model files present."
