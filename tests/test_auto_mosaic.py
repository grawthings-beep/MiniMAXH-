from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
import zipfile

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "custom_nodes" / "minimax_h3_ordered_storyboard" / "mosaic_nodes.py"


def load_mosaic_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = "/opt/ComfyUI/models"
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("minimax_mosaic_under_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_downloader():
    path = ROOT / "scripts" / "download_auto_mosaic.py"
    spec = importlib.util.spec_from_file_location("auto_mosaic_download_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoMosaicTests(unittest.TestCase):
    def setUp(self):
        self.module = load_mosaic_module()

    def test_default_contract_excludes_anus(self):
        required = self.module.WanAutoMosaicVideo.INPUT_TYPES()["required"]
        self.assertIs(required["enabled"][1]["default"], True)
        self.assertEqual(required["coverage_preset"][1]["default"], "JUST")
        self.assertEqual(required["confidence"][1]["default"], 0.30)
        self.assertEqual(required["iou_threshold"][1]["default"], 0.50)
        self.assertEqual(required["block_size"][1]["default"], 0)
        self.assertEqual(required["max_gap_frames"][1]["default"], 3)
        self.assertEqual(self.module.DEFAULT_CLASSES, "pussy,penis,testicles")
        self.assertNotIn("anus", self.module.DEFAULT_CLASSES)
        ids = self.module._selected_class_ids(
            dict(enumerate(self.module.MODEL_CLASSES)), self.module.DEFAULT_CLASSES
        )
        self.assertEqual(ids, [1, 3, 6])

    def test_gap_fill_wraps_loop_and_never_replaces_valid_contours(self):
        first = np.zeros((3, 3), dtype=np.bool_)
        third = np.zeros((3, 3), dtype=np.bool_)
        first[1, 0] = True
        third[1, 2] = True
        filled = self.module._fill_short_circular_gaps([first, None, third, None], 1)
        self.assertTrue(np.array_equal(filled[0], first))
        self.assertTrue(np.array_equal(filled[2], third))
        self.assertIsNotNone(filled[1])
        self.assertIsNotNone(filled[3])

    def test_fixed_grid_and_automatic_block_rule(self):
        image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        result = self.module._fixed_grid_mosaic(image, 3)
        self.assertTrue(np.all(result[0:3, 0:3] == result[0, 0]))
        self.assertFalse(np.array_equal(result[0, 0], result[3, 3]))
        self.assertEqual(self.module._resolve_block_size(0, 528, 704), 11)
        self.assertEqual(self.module._resolve_block_size(0, 320, 180), 10)

    def test_implementation_forces_ultralytics_and_tensors_to_cpu(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('device="cpu"', source)
        self.assertIn('half=False', source)
        self.assertIn('torch.empty_like(images, device="cpu")', source)
        self.assertNotIn('device="cuda"', source)

    def test_manifest_and_secure_resumable_downloader_contract(self):
        manifest = json.loads((ROOT / "manifests" / "auto_mosaic.json").read_text(encoding="utf-8"))
        item = manifest["files"][0]
        self.assertEqual(item["size_bytes"], 18_846_815)
        self.assertEqual(
            item["sha256"],
            "aca92864d30384b8dd7851b32e7ade621a147730bf9710fb4417214e0c61d690",
        )
        self.assertEqual(item["archive_member"], self.module.MODEL_FILENAME)
        self.assertEqual(item["provides"], [f"auto_mosaic/{self.module.MODEL_FILENAME}"])
        self.assertEqual(item["requires_env"], ["CIVITAI_API_TOKEN"])

        downloader = load_downloader()
        request = downloader.build_request(item["url"], "test-secret", 1024)
        self.assertNotIn("test-secret", request.full_url)
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer test-secret")
        self.assertEqual(request.headers["Range"], "bytes=1024-")
        redirected = downloader.urllib.request.HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://signed-storage.example/model.zip"
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_verified_zip_extracts_only_the_pinned_member(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("ntd11_anime_nsfw_segm_v5.pt", b"model-payload")
                bundle.writestr("ignore.txt", b"not extracted")
            info = downloader.verify_archive(
                archive, archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest()
            )
            self.assertEqual(info.filename, "ntd11_anime_nsfw_segm_v5.pt")
            destination = root / "models" / info.filename
            destination.parent.mkdir()
            downloader.extract_verified_member(archive, info.filename, destination)
            self.assertEqual(destination.read_bytes(), b"model-payload")
            self.assertFalse((destination.parent / "ignore.txt").exists())


if __name__ == "__main__":
    unittest.main()
