"""Offline safety checks for the imported FoxFur Seed Hunter profile."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "custom_nodes"))
from extract_foxhunter_rife import extract  # noqa: E402
from download_foxhunter_hf import download as download_private_hf, validate_asset  # noqa: E402
from download_foxhunter_models import public_manifest  # noqa: E402
from verify_foxhunter import validate  # noqa: E402
from foxhunter_compat import NODE_CLASS_MAPPINGS  # noqa: E402


class FoxHunterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads((ROOT / "workflows/foxhunter_seed_hunter_v3.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / "manifests/foxhunter_models.json").read_text(encoding="utf-8"))

    def test_workflow_and_model_contract(self):
        validate(self.workflow, self.manifest)

    def test_private_lora_is_excluded_from_public_parallel_manifest(self):
        public = public_manifest(self.manifest)
        self.assertEqual(len(public["files"]), 7)
        self.assertEqual(public["total_bytes"], sum(f["size"] for f in public["files"]))
        self.assertFalse(any(f.get("auth") == "hf" for f in public["files"]))
        self.assertTrue(any("10Eros_Max" in f["path"] for f in public["files"]))

    def test_private_lora_location_is_pinned_and_safe(self):
        asset = next(f for f in self.manifest["files"] if f.get("auth") == "hf")
        validate_asset(asset)
        for change in ({"path": "loras/../bad.safetensors"},
                       {"source_path": "../bad.safetensors"},
                       {"source_repo_id": "https://huggingface.co/bad"},
                       {"sha256": "0"}):
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                validate_asset({**asset, **change})

    def test_private_hf_lora_download_verifies_before_publish(self):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as temporary:
            root = Path(temporary)
            header = json.dumps({"test.weight": {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
            }}).encode()
            payload = len(header).to_bytes(8, "little") + header + b"\0" * 4
            cached = root / "cached.safetensors"
            cached.write_bytes(payload)
            asset = {
                "path": "loras/MINIMAX/test.safetensors", "auth": "hf",
                "source_repo_id": "example/test", "source_path": "test.safetensors",
                "source_revision": "main", "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            fake_hub = types.SimpleNamespace(
                HfApi=lambda **_: types.SimpleNamespace(
                    model_info=lambda *_args, **_kwargs: types.SimpleNamespace(sha="a" * 40)
                ),
                hf_hub_download=lambda **_: cached,
            )
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                download_private_hf(asset, root, "not-logged-token", 1)
            self.assertEqual((root / asset["path"]).read_bytes(), payload)
            self.assertFalse((root / "loras/MINIMAX/test.safetensors.part").exists())

    def test_small_easy_use_compat_nodes(self):
        self.assertEqual(NODE_CLASS_MAPPINGS["easy seed"]().doit(42), (42,))
        choice = NODE_CLASS_MAPPINGS["easy ifElse"]()
        self.assertEqual(choice.check_lazy_status(True, None, "other"), ["on_true"])
        self.assertEqual(choice.choose(True, "yes", "no"), ("yes",))
        self.assertEqual(choice.choose(False, "yes", "no"), ("no",))

    def test_missing_lora_is_rejected(self):
        workflow = copy.deepcopy(self.workflow)
        next(n for n in workflow["nodes"] if n["id"] == 554)["widgets_values"][0]["loras"][0]["name"] = "missing.safetensors"
        with self.assertRaises(AssertionError):
            validate(workflow, self.manifest)

    def test_missing_model_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"] = manifest["files"][:-1]
        manifest["total_bytes"] = sum(item["size"] for item in manifest["files"])
        with self.assertRaises(AssertionError):
            validate(self.workflow, manifest)

    def test_rife_extraction_uses_only_model_member(self):
        (ROOT / ".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp") as temporary:
            root = Path(temporary)
            archive = root / "rife.zip"
            model = b"R" * 1_000_001
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("RIFEv4.26_0921/train_log/flownet.pkl", model)
                bundle.writestr("../../should-not-extract.txt", "no")
            destination = root / "models/rife/flownet.pkl"
            extract(archive, destination)
            self.assertEqual(destination.read_bytes(), model)
            self.assertFalse((root / "should-not-extract.txt").exists())


if __name__ == "__main__":
    unittest.main()
