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
3. Snaps dimensions to the model grid (÷32) and duration to Wan's frame
   quantum (4k+1, max 121 frames ≈ 5s @ 24fps).
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
`ghcr.io/slzwei/farm-comfyui:v1`. After changing `runpod/` or `workflows/`,
run `bash scripts/sync-worker-mirror.sh` to propagate + rebuild.

## Models: network volume (chosen strategy)

Wan 2.2 TI2V-5B needs ~18GB of weights — too big to bake into the image
(every code tweak would re-push 27GB; registry cold pulls would dominate
startup) and far too big to download per invocation. A RunPod **network
volume** mounted at `/runpod-volume` gives one-time download, fast warm
loads, and lets the image stay slim.

```
/runpod-volume/models/
  diffusion_models/wan2.2_ti2v_5B_fp16.safetensors   (~10 GB)
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors (~6.7 GB)
  vae/wan2.2_vae.safetensors                          (~1.4 GB)
```

Populate it either way:

- **Self-provisioning (default)**: `start.sh` runs `scripts/ensure_models.sh`
  on boot — a fresh volume is filled automatically on the first cold start
  (~5-15 min, billed once; a mkdir lock stops concurrent workers from
  double-downloading). Set `SKIP_MODEL_CHECK=1` to bypass.
- **Manual**: create the volume (≥ 40GB) in the endpoint's datacenter,
  attach it to any temporary pod, run `scripts/download_models.sh`.

`extra_model_paths.yaml` points ComfyUI at the volume.

## Endpoint settings

- GPU: 24GB+ (L40S / 4090 / A40 class). fp16 5B fits comfortably.
- Workers: min 0, max 1 to start (scale after cost calibration).
- Network volume: attach the model volume.
- Execution timeout: ≥ 1800s (first render on a cold worker loads ~18GB of
  weights from the volume).
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
