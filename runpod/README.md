# farm-comfyui-worker

RunPod serverless worker that renders the farm's approved ComfyUI workflows.
Speaks **runpod-render-contract-v1** (mirrored in
`packages/shared/src/types/runpodContract.ts`). The end-to-end runbook lives
in `docs/RUNPOD_SETUP.md`; this file covers the worker itself.

## What it does

1. Validates the contract input (version, fields, product_image present).
2. Loads the bundled workflow for `workflow_id` — **never** a caller-supplied
   graph — and refuses version mismatches. Hash drift vs the farm's copy is
   logged and reported in metadata.
3. Snaps dimensions to the model grid (÷32) and duration to H3's frame
   quantum (17k+5, max 3600 frames — a full 18s brief renders in one pass).
4. Downloads the product image (size-capped, sha256'd) into ComfyUI's input
   dir under a job-scoped name.
5. Injects semantic inputs via the workflow's `_meta.injection` map.
6. Submits to local ComfyUI (`/prompt`), polls `/history` until done or
   `COMFY_TIMEOUT_SECONDS`.
7. Finds the output video (prefers the declared output node, scans all
   otherwise), requires non-empty, ffprobes it.
8. Uploads via the farm's presigned PUT (`video/mp4`), plus a
   `metadata.json` reproducibility record. Without upload URLs it falls back
   to inline base64 (≤ `MAX_BASE64_BYTES`, dev only).
9. Returns `{contract_version, videos[], metadata}`; failures return
   machine-readable `[CODE] message` errors.
10. Cleans up its input/output temp files (warm workers stay clean).

Error codes: `INPUT_INVALID`, `WORKFLOW_MISMATCH`, `ASSET_DOWNLOAD_FAILED`,
`COMFY_ERROR`, `COMFY_TIMEOUT`, `OUTPUT_MISSING`, `OUTPUT_INVALID`,
`UPLOAD_FAILED`. Signed URLs are never logged with their query strings.

## Image

- Base: `nvidia/cuda:12.8.1-runtime-ubuntu22.04`, Python 3.10 venv,
  torch 2.8.0+cu128, aria2, ffmpeg.
- **CUDA 12.8+ is mandatory for Blackwell** (RTX PRO 6000, RTX 50xx —
  sm_120). cu124 wheels carry no sm_120 kernels and die at runtime with
  "no kernel image is available for execution on the device". The same
  applies to the local RTX PRO 6000 box later.
- ComfyUI pinned via `--build-arg COMFYUI_VERSION=v0.34.0` (MiniMax H3
  nodes need ≥ v0.30.0; verify the tag exists before building and bump
  deliberately).
- **No custom nodes required** — the workflow uses core nodes only.
- Workflows are copied from the repo's `workflows/` at build time (single
  source of truth). Rebuild the image whenever a workflow file changes.

Build from the **repo root** (image ≈ 9GB; needs an x86_64 target):

```bash
docker buildx build --platform linux/amd64 \
  -f runpod/Dockerfile \
  -t <registry>/farm-comfyui-worker:v1 \
  --push .
```

No local Docker? `.github/workflows/build-worker.yml` builds and pushes
`ghcr.io/<owner>/farm-comfyui-worker:v1` on every push touching `runpod/` or
`workflows/` (or via workflow_dispatch).

**Production note (current deployment):** GHCR packages created from a
private repo are private, and RunPod pulls anonymously — so the deployed
image is built by the PUBLIC mirror repo `slzwei/farm-comfyui-worker`
(worker build inputs only, no farm code/secrets) as
`ghcr.io/slzwei/farm-comfyui:v2`. After changing `runpod/` or `workflows/`,
run `bash scripts/sync-worker-mirror.sh` to propagate + rebuild.

## Models: network volume (chosen strategy)

MiniMax H3 needs ~44.4GB of weights — far too big to bake into the image or
fetch per invocation. A RunPod **network volume** mounted at
`/runpod-volume` gives one-time download, fast warm loads, and keeps the
image slim.

```
/runpod-volume/models/
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors  (20.97 GB)
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors         (15.69 GB)
  vae/minimax_h3_video_vae_fp16.safetensors                          ( 5.21 GB)
  vae/minimax_h3_audio_vae_fp32.safetensors                          ( 0.61 GB)
  loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors    ( 1.96 GB)
```

Every file is verified against its exact byte size — a truncated multi-GB
checkpoint otherwise loads as noise or crashes ComfyUI.

Licence: MiniMax H3 Community Licence — commercial use permitted under
$20M revenue, **attribution required**.

Populate it either way:

- **Self-provisioning (default)**: `start.sh` runs `scripts/ensure_models.sh`
  on boot — a fresh volume is filled automatically on the first cold start
  (~5-15 min, billed once; a mkdir lock stops concurrent workers from
  double-downloading). Set `SKIP_MODEL_CHECK=1` to bypass.
- **Manual**: create the volume (≥ 40GB) in the endpoint's datacenter,
  attach it to any temporary pod, run `scripts/download_models.sh`.

`extra_model_paths.yaml` points ComfyUI at the volume.

## Endpoint settings

- GPU: **Blackwell RTX PRO 6000 (96GB) preferred** — H3 keeps ~37GB of
  weights resident, so 48GB cards leave little headroom for video
  activations. Any Blackwell card REQUIRES the cu128 image above.
- Workers: min 0, max 2 to start (scale after cost calibration).
- Network volume: attach the model volume (≥ 60GB for the H3 set).
- Execution timeout: ≥ 2700s (a cold worker loads ~44GB from the volume).
- Optional env overrides: `COMFY_TIMEOUT_SECONDS` (default 1500),
  `MAX_ASSET_BYTES`, `MAX_BASE64_BYTES`, `COMFY_EXTRA_ARGS`.

## Testing

Pure-logic tests (no GPU/network/deps):

```bash
python3 runpod/tests/test_handler.py    # or: pnpm test:worker
```

Local end-to-end (on any machine with the image + a GPU):

```bash
docker run --gpus all -p 8000:8000 -v /path/to/models:/runpod-volume \
  <registry>/farm-comfyui-worker:v1 \
  python -u /app/handler.py --rp_serve_api --rp_api_host 0.0.0.0
# then POST a contract payload to http://localhost:8000/run
```

Farm-side live verification: `pnpm verify:runpod` (see docs/RUNPOD_SETUP.md).
