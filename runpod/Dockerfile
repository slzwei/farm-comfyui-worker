# RunPod serverless worker: ComfyUI + farm handler (runpod-render-contract-v1).
#
# Build from the REPO ROOT so the approved workflow files are the single
# source of truth (no copies to drift):
#
#   docker buildx build --platform linux/amd64 \
#     -f runpod/Dockerfile -t <registry>/farm-comfyui-worker:v1 --push .
#
# Models are NOT baked in (~18GB for Wan 2.2 5B) — they live on a RunPod
# network volume mounted at /runpod-volume (see scripts/download_models.sh
# and extra_model_paths.yaml). Image stays ~9GB; cold start = container pull
# + model load from volume, not an 18GB download per worker.

# CUDA 12.8+ is REQUIRED for Blackwell (RTX PRO 6000 / RTX 50xx, sm_120).
# torch cu124 builds ship no sm_120 kernels and fail at runtime with
# "no kernel image is available for execution on the device".
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev \
      git curl aria2 ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch from the CUDA 12.8 wheel index — the first builds carrying sm_120
# (Blackwell) kernels.
#
# All THREE versions must be pinned as a matched set. Pinning torch alone
# lets pip resolve a torchaudio built against a different torch, and the ABI
# mismatch kills ComfyUI at import with
#   _torchaudio.abi3.so: undefined symbol: torch_library_impl
# ComfyUI imports torchaudio unconditionally (audio VAE), so this is fatal,
# not optional. torch 2.8.0 <-> torchvision 0.23.0 <-> torchaudio 2.8.0.
# The fallback installs all three unpinned TOGETHER so pip still resolves a
# self-consistent set.
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
 || pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128

# ComfyUI pinned by tag. MiniMax H3 nodes (MiniMaxH3ImageToVideo,
# VAEDecodeAudio, EmptyMiniMaxH3LatentAV) require >= v0.30.0; bump
# deliberately and re-verify.
ARG COMFYUI_VERSION=v0.34.0
RUN git clone --branch ${COMFYUI_VERSION} --depth 1 \
      https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && pip install -r /app/ComfyUI/requirements.txt

COPY runpod/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Approved workflows (bundled — the worker never accepts graphs from callers).
COPY workflows/ /app/workflows/
COPY runpod/extra_model_paths.yaml /app/ComfyUI/extra_model_paths.yaml
COPY runpod/handler.py runpod/start.sh /app/
COPY runpod/scripts/ /app/scripts/
RUN chmod +x /app/start.sh /app/scripts/*.sh

WORKDIR /app
ENV COMFY_HOST=127.0.0.1:8188 \
    WORKFLOWS_DIR=/app/workflows \
    COMFY_INPUT_DIR=/app/ComfyUI/input \
    COMFY_OUTPUT_DIR=/app/ComfyUI/output

CMD ["/app/start.sh"]
