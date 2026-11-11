"""Offline safety checks for the imported FoxFur Seed Hunter profile."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_foxhunter_rife import extract  # noqa: E402
from verify_foxhunter import validate  # noqa: E402


class FoxHunterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads((ROOT / "workflows/foxhunter_seed_hunter_v3.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / "manifests/foxhunter_models.json").read_text(encoding="utf-8"))

    def test_workflow_and_model_contract(self):
        validate(self.workflow, self.manifest)

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
