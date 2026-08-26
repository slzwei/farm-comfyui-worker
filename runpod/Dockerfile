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

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev \
      git curl aria2 ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch pinned to the CUDA 12.4 wheel index; keep in sync with the base image.
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124

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
