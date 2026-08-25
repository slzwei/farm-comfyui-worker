"""RunPod serverless worker for the farm's ComfyUI render pipeline.

Implements runpod-render-contract-v1 (see packages/shared/src/types/
runpodContract.ts — keep both sides in lockstep):

  input:  { contract_version, job_id, workflow_id, workflow_version,
            workflow_hash?, prompt, negative_prompt, width, height,
            duration_seconds, fps, seed, source_assets[{role,url}],
            parameters, output?{video_upload_url, metadata_upload_url?} }
  output: { contract_version, videos[{uploaded|base64, width, height,
            duration_seconds, size_bytes}], metadata{...} }
  errors: "[CODE] message" via RunPod's error channel. Codes: INPUT_INVALID,
          WORKFLOW_MISMATCH, ASSET_DOWNLOAD_FAILED, COMFY_ERROR,
          COMFY_TIMEOUT, OUTPUT_MISSING, OUTPUT_INVALID, UPLOAD_FAILED.

The worker owns the ComfyUI graph it bundles (workflows/<id>.json baked into
the image). Callers send semantic inputs only — arbitrary graphs are
impossible by construction. Node ids live exclusively in the workflow file's
_meta.injection map.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import random
import subprocess
import time
import urllib.parse
import uuid

import requests

CONTRACT_VERSION = "1"

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
WORKFLOWS_DIR = os.environ.get("WORKFLOWS_DIR", "/app/workflows")
COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", "/app/ComfyUI/input")
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/app/ComfyUI/output")
COMFY_TIMEOUT_SECONDS = int(os.environ.get("COMFY_TIMEOUT_SECONDS", "1500"))
MAX_ASSET_BYTES = int(os.environ.get("MAX_ASSET_BYTES", str(50 * 1024 * 1024)))
# RunPod caps status payloads (~20MB); stay safely below after +33% base64.
MAX_BASE64_BYTES = int(os.environ.get("MAX_BASE64_BYTES", str(12 * 1024 * 1024)))

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")
IMAGE_CONTENT_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class WorkerError(Exception):
    """Machine-readable failure: str(err) == '[CODE] message'."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def redact_url(url: str) -> str:
    """Signed URLs carry credentials in the query string — never log them."""
    try:
        p = urllib.parse.urlsplit(url)
        return f"{p.scheme}://{p.netloc}{p.path}"
    except ValueError:
        return "<unparseable-url>"


def log(message: str, **fields):
    payload = {"worker": "farm-comfyui", "msg": message}
    payload.update(fields)
    print(json.dumps(payload), flush=True)


# ---------------------------------------------------------------------------
# Input validation (mirror of RunPodWorkerInputSchema)
# ---------------------------------------------------------------------------

def validate_input(raw) -> dict:
    if not isinstance(raw, dict):
        raise WorkerError("INPUT_INVALID", "input must be an object")
    if raw.get("contract_version") != CONTRACT_VERSION:
        raise WorkerError(
            "INPUT_INVALID",
            f"unsupported contract_version {raw.get('contract_version')!r} (worker speaks {CONTRACT_VERSION})",
        )
    for field in ("job_id", "workflow_id", "workflow_version", "prompt"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise WorkerError("INPUT_INVALID", f"missing/empty field: {field}")
    for field in ("width", "height", "duration_seconds"):
        value = raw.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            raise WorkerError("INPUT_INVALID", f"field {field} must be a positive number")
    fps = raw.get("fps", 24)
    if not isinstance(fps, int) or fps <= 0:
        raise WorkerError("INPUT_INVALID", "fps must be a positive integer")
    seed = raw.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise WorkerError("INPUT_INVALID", "seed must be an integer or null")
    assets = raw.get("source_assets")
    if not isinstance(assets, list) or not assets:
        raise WorkerError("INPUT_INVALID", "source_assets must be a non-empty array")
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("url"), str)
            or not isinstance(asset.get("role"), str)
        ):
            raise WorkerError("INPUT_INVALID", "each source_asset needs role and url")
    if not any(a.get("role") == "product_image" for a in assets):
        raise WorkerError("INPUT_INVALID", "a product_image source asset is required")
    output = raw.get("output")
    if output is not None:
        if not isinstance(output, dict) or not isinstance(
            output.get("video_upload_url"), str
        ):
            raise WorkerError("INPUT_INVALID", "output.video_upload_url must be a string")
    return raw


# ---------------------------------------------------------------------------
# Workflow loading + semantic injection
# ---------------------------------------------------------------------------

def load_workflow(workflow_id: str, workflow_version: str, workflow_hash=None) -> dict:
    if "/" in workflow_id or "\\" in workflow_id or ".." in workflow_id:
        raise WorkerError("INPUT_INVALID", f"invalid workflow id {workflow_id!r}")
    path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
    if not os.path.isfile(path):
        raise WorkerError(
            "WORKFLOW_MISMATCH",
            f"workflow {workflow_id} is not bundled in this worker image",
        )
    with open(path, "rb") as f:
        raw = f.read()
    doc = json.loads(raw)
    meta = doc.get("_meta") or {}
    if meta.get("version") != workflow_version:
        raise WorkerError(
            "WORKFLOW_MISMATCH",
            f"caller wants {workflow_id}@{workflow_version} but image bundles @{meta.get('version')}",
        )
    bundled_hash = hashlib.sha256(raw).hexdigest()
    if workflow_hash and workflow_hash != bundled_hash:
        # Same version but different bytes: warn loudly, keep rendering — the
        # farm's hash lands in metadata so drift is visible in lineage.
        log(
            "workflow hash drift — rebuild the worker image",
            workflow_id=workflow_id,
            caller_hash=workflow_hash,
            bundled_hash=bundled_hash,
        )
    doc["_bundled_hash"] = bundled_hash
    return doc


def snap_dimension(value, multiple: int) -> int:
    """Floor to the model's grid (never exceeds requested size/VRAM)."""
    return max(multiple, (int(value) // multiple) * multiple)


def compute_frames(duration_seconds, fps: int, constraints: dict) -> int:
    quantum = int(constraints.get("frameQuantum", 1))
    offset = int(constraints.get("frameQuantumOffset", 0))
    min_frames = int(constraints.get("minFrames", 1))
    max_frames = int(constraints.get("maxFrames", 10_000))
    frames = round(duration_seconds * fps)
    if quantum > 1:
        frames = round((frames - offset) / quantum) * quantum + offset
    frames = max(min_frames, min(max_frames, frames))
    if quantum > 1 and (frames - offset) % quantum:
        frames -= (frames - offset) % quantum
    return frames


def inject_workflow_inputs(workflow_doc: dict, values: dict) -> dict:
    """Set semantic values into the graph per _meta.injection. Pure: returns a
    deep copy; every other node/field is untouched."""
    injection = (workflow_doc.get("_meta") or {}).get("injection") or {}
    graph = copy.deepcopy(workflow_doc["graph"])
    for semantic, value in values.items():
        target = injection.get(semantic)
        if not target:
            raise WorkerError(
                "WORKFLOW_MISMATCH", f"workflow has no injection target for {semantic!r}"
            )
        node = graph.get(target["node"])
        if node is None or target.get("input") is None:
            raise WorkerError(
                "WORKFLOW_MISMATCH",
                f"injection target for {semantic!r} points at missing node/input",
            )
        node["inputs"][target["input"]] = value
    return graph


def output_node_id(workflow_doc: dict):
    injection = (workflow_doc.get("_meta") or {}).get("injection") or {}
    target = injection.get("output") or {}
    return target.get("node")


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------

def download_asset(url: str, dest_dir: str, job_id: str) -> dict:
    os.makedirs(dest_dir, exist_ok=True)
    try:
        res = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as err:
        raise WorkerError(
            "ASSET_DOWNLOAD_FAILED", f"fetch failed for {redact_url(url)}: {err}"
        )
    if res.status_code == 403:
        raise WorkerError(
            "ASSET_DOWNLOAD_FAILED",
            f"{redact_url(url)} returned 403 — signed URL likely expired (raise S3_SIGNED_URL_TTL_SECONDS)",
        )
    if res.status_code != 200:
        raise WorkerError(
            "ASSET_DOWNLOAD_FAILED", f"{redact_url(url)} returned {res.status_code}"
        )
    content_type = (res.headers.get("content-type") or "").split(";")[0].strip()
    ext = IMAGE_CONTENT_EXT.get(content_type)
    if ext is None:
        guessed = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
        ext = guessed if guessed in (".png", ".jpg", ".jpeg", ".webp", ".gif") else ".png"
    filename = f"{job_id}-{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(dest_dir, filename)
    sha = hashlib.sha256()
    size = 0
    with open(path, "wb") as f:
        for chunk in res.iter_content(chunk_size=1 << 16):
            size += len(chunk)
            if size > MAX_ASSET_BYTES:
                f.close()
                os.remove(path)
                raise WorkerError(
                    "ASSET_DOWNLOAD_FAILED",
                    f"{redact_url(url)} exceeds {MAX_ASSET_BYTES} bytes",
                )
            sha.update(chunk)
            f.write(chunk)
    if size == 0:
        os.remove(path)
        raise WorkerError("ASSET_DOWNLOAD_FAILED", f"{redact_url(url)} was empty")
    return {"path": path, "filename": filename, "sha256": sha.hexdigest(), "size": size}


# ---------------------------------------------------------------------------
# ComfyUI driving
# ---------------------------------------------------------------------------

def comfy_url(pathname: str) -> str:
    return f"http://{COMFY_HOST}{pathname}"


def comfy_system_stats():
    try:
        res = requests.get(comfy_url("/system_stats"), timeout=10)
        return res.json() if res.status_code == 200 else None
    except requests.RequestException:
        return None


def submit_prompt(graph: dict) -> str:
    try:
        res = requests.post(
            comfy_url("/prompt"),
            json={"prompt": graph, "client_id": uuid.uuid4().hex},
            timeout=30,
        )
    except requests.RequestException as err:
        raise WorkerError("COMFY_ERROR", f"ComfyUI unreachable: {err}")
    if res.status_code != 200:
        detail = res.text[:600]
        raise WorkerError("COMFY_ERROR", f"/prompt rejected the graph: {detail}")
    prompt_id = res.json().get("prompt_id")
    if not prompt_id:
        raise WorkerError("COMFY_ERROR", "/prompt returned no prompt_id")
    return prompt_id


def wait_for_history(prompt_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            res = requests.get(comfy_url(f"/history/{prompt_id}"), timeout=30)
            if res.status_code == 200:
                entry = res.json().get(prompt_id)
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        messages = json.dumps(status.get("messages", []))[-800:]
                        raise WorkerError("COMFY_ERROR", f"execution failed: {messages}")
                    if status.get("completed") or entry.get("outputs"):
                        return entry
        except requests.RequestException:
            pass  # transient — ComfyUI busy under load; keep polling
        time.sleep(2)
    raise WorkerError(
        "COMFY_TIMEOUT", f"no completion within {timeout_seconds}s (COMFY_TIMEOUT_SECONDS)"
    )


def find_output_video(history_entry: dict, preferred_node) -> str:
    """Locate the rendered video file. Prefer the workflow's declared output
    node; fall back to scanning every node output (SaveVideo/VHS variants
    report under different keys: images/videos/gifs)."""
    outputs = history_entry.get("outputs") or {}
    node_ids = [preferred_node] if preferred_node in outputs else []
    node_ids += [n for n in outputs if n not in node_ids]
    for node_id in node_ids:
        for value in (outputs.get(node_id) or {}).values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                if not filename.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                path = os.path.join(
                    COMFY_OUTPUT_DIR, item.get("subfolder") or "", filename
                )
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    return path
    raise WorkerError(
        "OUTPUT_MISSING",
        "ComfyUI reported completion but no non-empty video file was found in its outputs",
    )


# ---------------------------------------------------------------------------
# Output validation + upload
# ---------------------------------------------------------------------------

def probe_video(path: str):
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if res.returncode != 0:
            return None
        parsed = json.loads(res.stdout)
        stream = (parsed.get("streams") or [{}])[0]
        duration = parsed.get("format", {}).get("duration")
        return {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "duration_seconds": float(duration) if duration else None,
        }
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return None  # ffprobe unavailable — size/existence checks still apply


def upload_put(url: str, path: str, content_type: str):
    last_error = None
    for attempt in range(3):
        try:
            with open(path, "rb") as f:
                res = requests.put(
                    url, data=f, headers={"Content-Type": content_type}, timeout=300
                )
            if 200 <= res.status_code < 300:
                return
            last_error = f"HTTP {res.status_code}: {res.text[:200]}"
            if res.status_code == 403:
                raise WorkerError(
                    "UPLOAD_FAILED",
                    f"upload to {redact_url(url)} returned 403 — presigned PUT likely expired",
                )
            if res.status_code < 500:
                break  # non-retryable
        except requests.RequestException as err:
            last_error = str(err)
        time.sleep(2 * (attempt + 1))
    raise WorkerError("UPLOAD_FAILED", f"upload to {redact_url(url)} failed: {last_error}")


def gpu_name():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return res.stdout.strip().splitlines()[0] if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def extract_model_info(graph: dict) -> dict:
    models = {}
    sampling = {}
    for node in graph.values():
        ct = node.get("class_type")
        inputs = node.get("inputs", {})
        if ct == "UNETLoader":
            models["diffusion"] = inputs.get("unet_name")
        elif ct == "CLIPLoader":
            models["text_encoder"] = inputs.get("clip_name")
        elif ct == "VAELoader":
            models["vae"] = inputs.get("vae_name")
        elif ct == "CheckpointLoaderSimple":
            models["checkpoint"] = inputs.get("ckpt_name")
        elif ct == "LoraLoader":
            models.setdefault("loras", []).append(
                {"name": inputs.get("lora_name"), "strength": inputs.get("strength_model")}
            )
        elif ct == "KSampler":
            sampling.update(
                sampler=inputs.get("sampler_name"),
                scheduler=inputs.get("scheduler"),
                steps=inputs.get("steps"),
                cfg=inputs.get("cfg"),
            )
        elif ct == "ModelSamplingSD3":
            sampling["shift"] = inputs.get("shift")
    return {"models": models, **sampling}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event):
    t_start = time.time()
    timings = {}
    cleanup_paths = []
    try:
        inp = validate_input(event.get("input"))
        job_id = inp["job_id"]
        log("job received", job_id=job_id, workflow_id=inp["workflow_id"])

        workflow_doc = load_workflow(
            inp["workflow_id"], inp["workflow_version"], inp.get("workflow_hash")
        )
        constraints = (workflow_doc.get("_meta") or {}).get("constraints") or {}
        multiple = int(constraints.get("dimensionMultiple", 1))
        fps = int(inp.get("fps", 24))
        width = snap_dimension(inp["width"], multiple)
        height = snap_dimension(inp["height"], multiple)
        frames = compute_frames(inp["duration_seconds"], fps, constraints)
        seed = inp.get("seed")
        if seed is None:
            seed = random.randrange(2**31)

        # Source assets → ComfyUI input dir.
        t0 = time.time()
        product = next(a for a in inp["source_assets"] if a["role"] == "product_image")
        asset = download_asset(product["url"], COMFY_INPUT_DIR, job_id)
        cleanup_paths.append(asset["path"])
        timings["asset_download_seconds"] = round(time.time() - t0, 3)

        graph = inject_workflow_inputs(
            workflow_doc,
            {
                "product_image": asset["filename"],
                "prompt": inp["prompt"],
                "negative_prompt": inp.get("negative_prompt", ""),
                "seed": seed,
                "width": width,
                "height": height,
                "frame_count": frames,
                "fps": fps,
            },
        )

        t0 = time.time()
        prompt_id = submit_prompt(graph)
        log("comfy prompt queued", job_id=job_id, prompt_id=prompt_id, frames=frames,
            width=width, height=height, seed=seed)
        history = wait_for_history(prompt_id, COMFY_TIMEOUT_SECONDS)
        timings["render_seconds"] = round(time.time() - t0, 3)

        video_path = find_output_video(history, output_node_id(workflow_doc))
        cleanup_paths.append(video_path)
        size_bytes = os.path.getsize(video_path)
        probe = probe_video(video_path)
        if probe is not None and not probe.get("duration_seconds"):
            raise WorkerError(
                "OUTPUT_INVALID", "rendered file exists but ffprobe reads no duration"
            )

        stats = comfy_system_stats() or {}
        metadata = {
            "contract_version": CONTRACT_VERSION,
            "job_id": job_id,
            "workflow_id": inp["workflow_id"],
            "workflow_version": inp["workflow_version"],
            "workflow_hash": workflow_doc.get("_bundled_hash"),
            "caller_workflow_hash": inp.get("workflow_hash"),
            "seed": seed,
            "frames": frames,
            "fps": fps,
            "width": (probe or {}).get("width") or width,
            "height": (probe or {}).get("height") or height,
            "duration_seconds": (probe or {}).get("duration_seconds") or frames / fps,
            "requested": {
                "width": inp["width"], "height": inp["height"],
                "duration_seconds": inp["duration_seconds"],
            },
            "gpu_name": gpu_name(),
            "comfyui_version": (stats.get("system") or {}).get("comfyui_version"),
            "input_assets": [{"role": "product_image", "sha256": asset["sha256"]}],
            "timings": timings,
            **extract_model_info(graph),
        }

        video_entry = {
            "width": metadata["width"],
            "height": metadata["height"],
            "duration_seconds": metadata["duration_seconds"],
            "size_bytes": size_bytes,
        }
        output_cfg = inp.get("output")
        if output_cfg and output_cfg.get("video_upload_url"):
            t0 = time.time()
            upload_put(output_cfg["video_upload_url"], video_path, "video/mp4")
            video_entry["uploaded"] = True
            if output_cfg.get("metadata_upload_url"):
                meta_path = video_path + ".metadata.json"
                cleanup_paths.append(meta_path)
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                try:
                    upload_put(
                        output_cfg["metadata_upload_url"], meta_path, "application/json"
                    )
                except WorkerError as err:
                    log("metadata upload failed (non-fatal)", job_id=job_id, error=str(err))
            timings["upload_seconds"] = round(time.time() - t0, 3)
        else:
            if size_bytes > MAX_BASE64_BYTES:
                raise WorkerError(
                    "UPLOAD_FAILED",
                    f"no upload URL and video ({size_bytes}B) exceeds the {MAX_BASE64_BYTES}B inline cap — configure S3/R2 storage",
                )
            with open(video_path, "rb") as f:
                video_entry["base64"] = base64.b64encode(f.read()).decode("ascii")

        timings["total_seconds"] = round(time.time() - t_start, 3)
        log("job complete", job_id=job_id, size_bytes=size_bytes, timings=timings)
        return {
            "contract_version": CONTRACT_VERSION,
            "videos": [video_entry],
            "metadata": metadata,
        }
    except WorkerError as err:
        log("job failed", error=str(err))
        return {"error": str(err)}
    except Exception as err:  # noqa: BLE001 — surface unexpected crashes with a code too
        log("job crashed", error=repr(err))
        return {"error": f"[COMFY_ERROR] unexpected worker crash: {err!r}"}
    finally:
        for path in cleanup_paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
