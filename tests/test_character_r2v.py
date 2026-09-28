from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


character = load("character_test", "custom_nodes/minimax_h3_ordered_storyboard/character_nodes.py")
manifest_tools = load("character_manifest_test", "scripts/prepare_character_manifest.py")


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads((ROOT / "workflows/character_reveal_r2v_2x.json").read_text(encoding="utf-8"))

    def test_durations_and_prompt_modes(self):
        self.assertEqual(character.aligned_frames(5), 124)
        self.assertEqual(character.aligned_frames(7), 175)
        self.assertEqual(character.aligned_frames(10), 243)
        self.assertEqual(character.aligned_frames(15), 362)
        for value in (0, 30, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                character.aligned_frames(value)
        prompt, length = character.MiniMaxH3CharacterPrompt().build(
            7, "direction", "Keep the original eyes.", "An eye peeks through a door.", "Footsteps.", "N/A"
        )
        self.assertEqual(length, 175)
        self.assertIn("7.29-second", prompt)
        self.assertIn("not as the first frame", prompt)
        self.assertIn("Keep the original eyes.", prompt)
        self.assertIn("non_diegetic_music:\nN/A", prompt)
        self.assertNotIn("<Picture 2>", prompt)
        raw = "  subject_definitions:\nMy complete custom prompt.\n"
        self.assertEqual(character.MiniMaxH3CharacterPrompt().build(5, "full_prompt", "ignored", raw, "ignored", "ignored"), (raw, 124))
        with self.assertRaises(ValueError):
            character.MiniMaxH3CharacterPrompt().build(5, "direction", "", " ", "", "")

    def test_reference_is_bounded_downscale_only(self):
        for width, height in ((2048, 3072), (4096, 2048), (512, 512), (640, 1024), (64, 8192)):
            w, h = character.reference_dimensions(width, height, 1024)
            self.assertLessEqual(w, width)
            self.assertLessEqual(h, height)
            self.assertLessEqual(max(w, h), 1024)
            self.assertEqual(w % 32, 0)
            self.assertEqual(h % 32, 0)
        self.assertEqual(character.reference_dimensions(512, 512, 1024), (512, 512))
        with self.assertRaises(ValueError):
            character.reference_dimensions(10, 512, 1024)
        image = mock.Mock(shape=(2, 512, 512, 3))
        with self.assertRaises(ValueError):
            character.MiniMaxH3CharacterReference().prepare(image, 1024)

    def test_registration_is_exported(self):
        self.assertEqual(set(character.NODE_CLASS_MAPPINGS), {"MiniMaxH3CharacterPrompt", "MiniMaxH3CharacterReference"})
        source = (ROOT / "custom_nodes/minimax_h3_ordered_storyboard/__init__.py").read_text(encoding="utf-8")
        self.assertIn("NODE_CLASS_MAPPINGS.update(CHARACTER_NODE_CLASS_MAPPINGS)", source)

    def test_manifest_merge_deduplicates_and_does_not_mutate(self):
        base = json.loads((ROOT / "manifests/minimax_h3_i2v_upscale.json").read_text())
        ref = json.loads((ROOT / "manifests/minimax_h3_r2v_upscale.json").read_text())
        original = copy.deepcopy(base)
        merged = manifest_tools.merge_manifests(base, ref)
        self.assertEqual(len(merged["files"]), 6)
        self.assertEqual(merged["total_bytes"], 63_508_026_812)
        self.assertEqual(base, original)
        self.assertEqual(manifest_tools.merge_manifests(merged, ref), merged)
        bad = copy.deepcopy(base)
        bad["revision"] = "main"
        with self.assertRaises(ValueError):
            manifest_tools.merge_manifests(bad, ref)
        bad = copy.deepcopy(ref)
        bad["files"][1]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            manifest_tools.merge_manifests(base, bad)

    def test_graph_has_one_appearance_reference_and_no_first_frame_or_lora(self):
        nodes = self.workflow["nodes"]
        types = [n["type"] for n in nodes]
        self.assertNotIn("definitions", self.workflow)
        for forbidden in ("MiniMaxH3ImageToVideo", "LoraLoaderModelOnly", "EasyCache", "MiniMaxH3TurboProfile", "LoadVideo", "ApplyMiniMaxH3FirstBlockCache"):
            self.assertNotIn(forbidden, types)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("WanAutoMosaicVideo"), 1)
        ref = next(n for n in nodes if n["type"] == "MiniMaxH3ReferenceToVideo")
        self.assertEqual(ref["widgets_values"][-1], "max")
        connected = [s["name"] for s in ref["inputs"] if s.get("link") is not None and s["name"].startswith("ref_")]
        self.assertEqual(connected, ["ref_images.ref_image_0"])
        self.assertEqual(next(n for n in nodes if n["type"] == "BasicScheduler")["widgets_values"], ["normal", 25, 1.0])
        self.assertEqual(next(n for n in nodes if n["type"] == "ResolutionSelector")["widgets_values"], ["9:16 (Portrait Widescreen)", 0.4, 32])

    def test_bidirectional_links_types_and_topological_order(self):
        by_id = {n["id"]: n for n in self.workflow["nodes"]}
        links = {l[0]: l for l in self.workflow["links"]}
        self.assertEqual(len(links), len(self.workflow["links"]))
        for lid, a, ai, b, bi, type_ in links.values():
            self.assertIn(lid, by_id[a]["outputs"][ai]["links"])
            self.assertEqual(lid, by_id[b]["inputs"][bi]["link"])
            self.assertEqual(type_, by_id[a]["outputs"][ai]["type"])
            self.assertEqual(type_, by_id[b]["inputs"][bi]["type"])
            self.assertLess(by_id[a]["order"], by_id[b]["order"])
        for node in by_id.values():
            for index, socket in enumerate(node.get("inputs", [])):
                if socket.get("link") is not None:
                    self.assertEqual(links[socket["link"]][3:5], [node["id"], index])
            for index, socket in enumerate(node.get("outputs", [])):
                for lid in socket.get("links") or []:
                    self.assertEqual(links[lid][1:3], [node["id"], index])

    def test_no_node_or_group_overlap_and_every_node_is_contained(self):
        def overlaps(a, b):
            x, y, w, h = a
            u, v, p, q = b
            return max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q)
        nodes = self.workflow["nodes"]
        boxes = [n["pos"] + n["size"] for n in nodes]
        groups = [g["bounding"] for g in self.workflow["groups"]]
        for collection in (boxes, groups):
            for i, a in enumerate(collection):
                for b in collection[i+1:]:
                    self.assertFalse(overlaps(a, b), (a, b))
        for x, y, w, h in boxes:
            self.assertEqual(sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh for gx, gy, gw, gh in groups), 1)

    def test_generated_workflow_and_existing_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/build_character_workflow.py"), "--output-dir", directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(directory) / "character_reveal_r2v_2x.json").read_text(encoding="utf-8")), self.workflow)
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/verify_workflow.py"),
            "--workflow", str(ROOT / "workflows/character_reveal_r2v_2x.json"),
            "--manifest", str(ROOT / "manifests/minimax_h3_r2v_upscale.json"),
            "--mode", "r2v", "--expect-upscale", "--expect-memory-safe-decode", "--expect-auto-mosaic",
            "--auto-mosaic-manifest", str(ROOT / "manifests/auto_mosaic.json"),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_is_opt_in_and_checks_before_launch(self):
        entrypoint = (ROOT / "scripts/entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('CHARACTER_R2V="${H3_CHARACTER_R2V:-0}"', entrypoint)
        merge = entrypoint.index('"${SCRIPT_DIR}/prepare_character_manifest.py"')
        preflight = entrypoint.index('python "${SCRIPT_DIR}/preflight.py"')
        self.assertLess(merge, preflight)
        self.assertIn('export MODEL_MANIFEST="${MANIFEST}"', entrypoint)
        self.assertLess(entrypoint.index('python "${SCRIPT_DIR}/verify_models.py"'), entrypoint.index('exec python main.py'))
        docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("H3_CHARACTER_R2V=1", docker)
        self.assertIn("H3_CHARACTER_R2V=0", docker)


if __name__ == "__main__":
    unittest.main()
