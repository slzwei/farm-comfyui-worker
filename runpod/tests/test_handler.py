#!/usr/bin/env python3
"""Stdlib-only tests for the worker's pure logic (validation, injection,
frame/dimension snapping, output discovery). No pytest, no network, no GPU:

    python3 runpod/tests/test_handler.py

The injection tests load the REAL workflows/product-demo-v1.json from the
repo, so the worker-side contract is proven against the exact file the image
bundles."""

import json
import os
import sys
import tempfile
import types
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# The pure functions under test never touch the network — stub `requests` so
# the tests run on machines without it installed.
sys.modules.setdefault("requests", types.ModuleType("requests"))
os.environ.setdefault("WORKFLOWS_DIR", os.path.join(REPO_ROOT, "workflows"))
sys.path.insert(0, os.path.join(REPO_ROOT, "runpod"))

import handler  # noqa: E402


def valid_input(**overrides):
    base = {
        "contract_version": "1",
        "job_id": "genjob_1",
        "workflow_id": "product-demo-v1",
        "workflow_version": "2",
        "prompt": "a product demo",
        "negative_prompt": "",
        "width": 704,
        "height": 1280,
        "duration_seconds": 5,
        "fps": 24,
        "seed": 42,
        "source_assets": [{"role": "product_image", "url": "https://x/img.png"}],
        "parameters": {},
    }
    base.update(overrides)
    return base


class TestValidation(unittest.TestCase):
    def test_accepts_valid_input(self):
        handler.validate_input(valid_input())

    def assert_rejected(self, code, **overrides):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.validate_input(valid_input(**overrides))
        self.assertEqual(ctx.exception.code, code)

    def test_rejects_bad_contract_version(self):
        self.assert_rejected("INPUT_INVALID", contract_version="2")

    def test_rejects_missing_prompt(self):
        self.assert_rejected("INPUT_INVALID", prompt="")

    def test_rejects_nonpositive_dimensions(self):
        self.assert_rejected("INPUT_INVALID", width=0)

    def test_rejects_missing_product_image(self):
        self.assert_rejected(
            "INPUT_INVALID", source_assets=[{"role": "reference", "url": "https://x"}]
        )

    def test_rejects_fractional_fps(self):
        self.assert_rejected("INPUT_INVALID", fps=23.976)

    def test_rejects_malformed_output_block(self):
        self.assert_rejected("INPUT_INVALID", output={"nope": True})


class TestWorkflowLoading(unittest.TestCase):
    def test_loads_bundled_workflow_with_hash(self):
        doc = handler.load_workflow("product-demo-v1", "2")
        self.assertIn("graph", doc)
        self.assertEqual(len(doc["_bundled_hash"]), 64)

    def test_version_mismatch_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("product-demo-v1", "99")
        self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")

    def test_unknown_workflow_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("not-a-workflow", "2")
        self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")

    def test_traversal_in_workflow_id_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("../secrets", "2")
        self.assertEqual(ctx.exception.code, "INPUT_INVALID")


class TestSnapping(unittest.TestCase):
    def test_dimensions_floor_to_grid(self):
        self.assertEqual(handler.snap_dimension(720, 32), 704)
        self.assertEqual(handler.snap_dimension(1280, 32), 1280)
        self.assertEqual(handler.snap_dimension(31, 32), 32)  # never below one unit

    def test_frames_snap_to_h3_quantum(self):
        constraints = {
            "frameQuantum": 17, "frameQuantumOffset": 5,
            "minFrames": 5, "maxFrames": 3600,
        }
        # 5s @ 24fps = 120 frames -> nearest 17k+5 = 124 (the template default)
        self.assertEqual(handler.compute_frames(5, 24, constraints), 124)
        # A full 18s brief now fits in ONE pass — the reason for moving off Wan.
        self.assertEqual(handler.compute_frames(18, 24, constraints), 430)
        # tiny durations clamp up to the minimum, still on the quantum
        f = handler.compute_frames(0.1, 24, constraints)
        self.assertGreaterEqual(f, 5)
        self.assertEqual((f - 5) % 17, 0)
        # every produced count sits on the quantum
        for seconds in (1, 2, 3, 4, 5, 7, 10, 18, 30):
            f = handler.compute_frames(seconds, 24, constraints)
            self.assertEqual((f - 5) % 17, 0, f"{seconds}s -> {f}")
            self.assertLessEqual(f, 3600)


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.doc = handler.load_workflow("product-demo-v1", "2")
        self.injection = self.doc["_meta"]["injection"]

    def inject(self):
        return handler.inject_workflow_inputs(
            self.doc,
            {
                "product_image": "genjob_1-abc.png",
                "prompt": "hero shot of the bottle",
                "negative_prompt": "blurry",  # unsupported by H3 — must be skipped
                "seed": 1234,
                "width": 704,
                "height": 1280,
                "frame_count": 124,
                "fps": 24,
            },
        )

    def node(self, graph, semantic):
        return graph[self.injection[semantic]["node"]]

    def test_semantic_values_land_on_mapped_nodes(self):
        graph = self.inject()
        self.assertEqual(self.node(graph, "product_image")["inputs"]["image"], "genjob_1-abc.png")
        self.assertEqual(self.node(graph, "prompt")["inputs"]["prompt"], "hero shot of the bottle")
        self.assertEqual(self.node(graph, "seed")["inputs"]["noise_seed"], 1234)
        self.assertEqual(self.node(graph, "width")["inputs"]["width"], 704)
        self.assertEqual(self.node(graph, "frame_count")["inputs"]["length"], 124)
        self.assertEqual(self.node(graph, "fps")["inputs"]["fps"], 24)

    def test_unsupported_semantic_is_skipped_not_fatal(self):
        # H3 has no negative-prompt input; injecting one must not raise and
        # must not invent a node.
        graph = self.inject()
        self.assertNotIn("negative_prompt", self.injection)
        self.assertEqual(set(graph.keys()), set(self.doc["graph"].keys()))

    def test_rest_of_graph_untouched(self):
        graph = self.inject()
        info = handler.extract_model_info(graph)
        self.assertEqual(info["steps"], 6)
        self.assertEqual(info["sampler"], "res_multistep")
        self.assertEqual(info["scheduler"], "simple")
        # links survive injection
        i2v = self.node(graph, "prompt")
        self.assertEqual(i2v["inputs"]["clip"], ["2", 0])
        self.assertEqual(i2v["inputs"]["vae"], ["3", 0])
        # node count unchanged — injection never adds/removes nodes
        self.assertEqual(set(graph.keys()), set(self.doc["graph"].keys()))

    def test_injection_is_pure(self):
        before = json.dumps(self.doc["graph"], sort_keys=True)
        self.inject()
        self.assertEqual(json.dumps(self.doc["graph"], sort_keys=True), before)

    def test_unmapped_semantic_is_ignored(self):
        # Unknown/unsupported semantics are skipped rather than fatal, so one
        # graph can serve models with different input surfaces.
        graph = handler.inject_workflow_inputs(self.doc, {"lora_strength": 1})
        self.assertEqual(set(graph.keys()), set(self.doc["graph"].keys()))

    def test_workflow_missing_a_required_semantic_is_rejected(self):
        # The safety net that replaces per-injection strictness: a workflow
        # whose map omits a REQUIRED semantic must fail to load, so a typo
        # can never silently render with baked-in defaults.
        import copy as _copy
        broken = _copy.deepcopy(self.doc)
        del broken["_meta"]["injection"]["seed"]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "broken-wf.json"), "w") as f:
                json.dump(broken, f)
            old_dir = handler.WORKFLOWS_DIR
            handler.WORKFLOWS_DIR = tmp
            try:
                with self.assertRaises(handler.WorkerError) as ctx:
                    handler.load_workflow("broken-wf", "2")
                self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")
                self.assertIn("seed", str(ctx.exception))
            finally:
                handler.WORKFLOWS_DIR = old_dir


class TestOutputDiscovery(unittest.TestCase):
    def run_with_outputs(self, outputs, files):
        with tempfile.TemporaryDirectory() as tmp:
            old = handler.COMFY_OUTPUT_DIR
            handler.COMFY_OUTPUT_DIR = tmp
            try:
                for rel, content in files.items():
                    path = os.path.join(tmp, rel)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)
                return handler.find_output_video({"outputs": outputs}, "12")
            finally:
                handler.COMFY_OUTPUT_DIR = old

    def test_finds_video_on_preferred_node(self):
        path = self.run_with_outputs(
            {"12": {"images": [{"filename": "clip.mp4", "subfolder": "farm", "type": "output"}]}},
            {"farm/clip.mp4": b"videobytes"},
        )
        self.assertTrue(path.endswith("farm/clip.mp4"))

    def test_scans_other_nodes_and_keys(self):
        path = self.run_with_outputs(
            {"99": {"gifs": [{"filename": "out.mp4", "subfolder": ""}]}},
            {"out.mp4": b"x"},
        )
        self.assertTrue(path.endswith("out.mp4"))

    def test_empty_file_is_missing(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            self.run_with_outputs(
                {"12": {"images": [{"filename": "clip.mp4", "subfolder": ""}]}},
                {"clip.mp4": b""},
            )
        self.assertEqual(ctx.exception.code, "OUTPUT_MISSING")

    def test_non_video_outputs_are_ignored(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            self.run_with_outputs(
                {"12": {"images": [{"filename": "frame.png", "subfolder": ""}]}},
                {"frame.png": b"png"},
            )
        self.assertEqual(ctx.exception.code, "OUTPUT_MISSING")


class TestMisc(unittest.TestCase):
    def test_error_string_carries_machine_code(self):
        self.assertEqual(
            str(handler.WorkerError("OUTPUT_MISSING", "no video")),
            "[OUTPUT_MISSING] no video",
        )

    def test_redact_url_strips_signed_query(self):
        self.assertEqual(
            handler.redact_url("https://acc.r2.dev/clips/j/clip.mp4?X-Amz-Signature=SECRET"),
            "https://acc.r2.dev/clips/j/clip.mp4",
        )

    def test_extract_model_info_reads_the_real_graph(self):
        doc = handler.load_workflow("product-demo-v1", "2")
        info = handler.extract_model_info(doc["graph"])
        self.assertEqual(
            info["models"]["diffusion"], "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        )
        self.assertEqual(
            info["models"]["text_encoder"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        )
        self.assertEqual(info["models"]["vae"], "minimax_h3_video_vae_fp16.safetensors")
        # audio VAE is recorded too — H3 generates native sound
        self.assertIn("minimax_h3_audio_vae_fp32.safetensors", info["models"]["vae_extra"])
        self.assertEqual(info["models"]["loras"][0]["strength"], 1.0)
        self.assertEqual(info["sampler"], "res_multistep")
        self.assertEqual(info["steps"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestModelManifest(unittest.TestCase):
    """The size floors in ensure_models.sh must equal the real file sizes.

    A floor even slightly ABOVE the true size makes every worker treat a
    complete download as truncated, wipe it, re-fetch tens of GB and die
    (observed 26 Aug 2026). Network-free: parses the script and compares
    against sizes captured from the HuggingFace tree API.
    """

    # Exact bytes from:
    #   https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/<dir>
    #   https://huggingface.co/api/models/lightx2v/Minimax-h3-Turbo/tree/main
    EXPECTED = {
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20970379616,
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15687142551,
        "vae/minimax_h3_video_vae_fp16.safetensors": 5207808496,
        "vae/minimax_h3_audio_vae_fp32.safetensors": 605254808,
        "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": 1956193000,
    }

    def test_size_floors_match_real_file_sizes(self):
        import re
        script = os.path.join(REPO_ROOT, "runpod", "scripts", "ensure_models.sh")
        with open(script) as f:
            body = f.read()
        for path, size in self.EXPECTED.items():
            m = re.search(rf'"{re.escape(path)}\|(\d+)"', body)
            self.assertIsNotNone(m, f"{path} missing from ensure_models.sh FILES")
            baked = int(m.group(1))
            self.assertLessEqual(
                baked, size,
                f"{path}: floor {baked} exceeds real size {size} — every worker "
                f"would reject a complete download and re-fetch forever",
            )
            self.assertEqual(baked, size, f"{path}: floor should be the exact size")
