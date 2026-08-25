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
        "workflow_version": "1",
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
        doc = handler.load_workflow("product-demo-v1", "1")
        self.assertIn("graph", doc)
        self.assertEqual(len(doc["_bundled_hash"]), 64)

    def test_version_mismatch_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("product-demo-v1", "99")
        self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")

    def test_unknown_workflow_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("not-a-workflow", "1")
        self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")

    def test_traversal_in_workflow_id_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.load_workflow("../secrets", "1")
        self.assertEqual(ctx.exception.code, "INPUT_INVALID")


class TestSnapping(unittest.TestCase):
    def test_dimensions_floor_to_grid(self):
        self.assertEqual(handler.snap_dimension(720, 32), 704)
        self.assertEqual(handler.snap_dimension(1280, 32), 1280)
        self.assertEqual(handler.snap_dimension(31, 32), 32)  # never below one unit

    def test_frames_snap_to_wan_quantum(self):
        constraints = {
            "frameQuantum": 4, "frameQuantumOffset": 1,
            "minFrames": 9, "maxFrames": 121,
        }
        # 5s @ 24fps = 120 frames -> nearest 4k+1 = 121
        self.assertEqual(handler.compute_frames(5, 24, constraints), 121)
        # 18s brief clamps to the model ceiling
        self.assertEqual(handler.compute_frames(18, 24, constraints), 121)
        # tiny durations clamp up to the minimum, still on the quantum
        f = handler.compute_frames(0.1, 24, constraints)
        self.assertGreaterEqual(f, 9)
        self.assertEqual((f - 1) % 4, 0)
        # every produced count sits on the quantum
        for seconds in (1, 2, 3, 4, 5, 7, 10, 18):
            f = handler.compute_frames(seconds, 24, constraints)
            self.assertEqual((f - 1) % 4, 0, f"{seconds}s -> {f}")
            self.assertLessEqual(f, 121)


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.doc = handler.load_workflow("product-demo-v1", "1")
        self.injection = self.doc["_meta"]["injection"]

    def inject(self):
        return handler.inject_workflow_inputs(
            self.doc,
            {
                "product_image": "genjob_1-abc.png",
                "prompt": "hero shot of the bottle",
                "negative_prompt": "blurry",
                "seed": 1234,
                "width": 704,
                "height": 1280,
                "frame_count": 121,
                "fps": 24,
            },
        )

    def node(self, graph, semantic):
        return graph[self.injection[semantic]["node"]]

    def test_semantic_values_land_on_mapped_nodes(self):
        graph = self.inject()
        self.assertEqual(self.node(graph, "product_image")["inputs"]["image"], "genjob_1-abc.png")
        self.assertEqual(self.node(graph, "prompt")["inputs"]["text"], "hero shot of the bottle")
        self.assertEqual(self.node(graph, "negative_prompt")["inputs"]["text"], "blurry")
        self.assertEqual(self.node(graph, "seed")["inputs"]["seed"], 1234)
        self.assertEqual(self.node(graph, "width")["inputs"]["width"], 704)
        self.assertEqual(self.node(graph, "frame_count")["inputs"]["length"], 121)
        self.assertEqual(self.node(graph, "fps")["inputs"]["fps"], 24)

    def test_rest_of_graph_untouched(self):
        graph = self.inject()
        sampler = self.node(graph, "seed")
        self.assertEqual(sampler["inputs"]["steps"], 30)
        self.assertEqual(sampler["inputs"]["cfg"], 5.0)
        self.assertEqual(sampler["inputs"]["sampler_name"], "uni_pc")
        # links survive injection
        self.assertEqual(sampler["inputs"]["model"], ["7", 0])
        # node count unchanged — injection never adds/removes nodes
        self.assertEqual(set(graph.keys()), set(self.doc["graph"].keys()))

    def test_injection_is_pure(self):
        before = json.dumps(self.doc["graph"], sort_keys=True)
        self.inject()
        self.assertEqual(json.dumps(self.doc["graph"], sort_keys=True), before)

    def test_unknown_semantic_fails(self):
        with self.assertRaises(handler.WorkerError) as ctx:
            handler.inject_workflow_inputs(self.doc, {"lora_strength": 1})
        self.assertEqual(ctx.exception.code, "WORKFLOW_MISMATCH")


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
        doc = handler.load_workflow("product-demo-v1", "1")
        info = handler.extract_model_info(doc["graph"])
        self.assertEqual(info["models"]["diffusion"], "wan2.2_ti2v_5B_fp16.safetensors")
        self.assertEqual(info["models"]["vae"], "wan2.2_vae.safetensors")
        self.assertEqual(info["sampler"], "uni_pc")
        self.assertEqual(info["steps"], 30)
        self.assertEqual(info["shift"], 8.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
