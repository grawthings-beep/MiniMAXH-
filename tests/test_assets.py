from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "minimax_h3_i2v.json"
WORKFLOW = ROOT / "workflows" / "minimax_h3_i2v.json"


class AssetTests(unittest.TestCase):
    def test_manifest_is_exact_i2v_set(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        files = manifest["files"]
        paths = {item["path"] for item in files}
        self.assertEqual(manifest["total_bytes"], sum(item["size"] for item in files))
        self.assertEqual(len(paths), 4)
        self.assertIn(
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            paths,
        )
        self.assertFalse(any("ref2va" in path for path in paths))
        for item in files:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(item["size"], 0)

    def test_workflow_is_pinned_upstream_asset(self) -> None:
        digest = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
        self.assertEqual(
            digest, "bb71aecdd3c0b62e56eafe03acb14d1cfeabec7072eaed9cbdf473c2aaf73009"
        )
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        serialized = json.dumps(workflow)
        self.assertIn("minimax_h3_fl2va_pruned_int8_convrot.safetensors", serialized)
        self.assertNotIn("minimax_h3_ref2va", serialized)
        self.assertIn("MiniMaxH3ImageToVideo", serialized)

    def test_dockerfile_pins_comfyui(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG COMFYUI_VERSION=v0.30.0", dockerfile)
        self.assertRegex(dockerfile, r"ARG COMFYUI_COMMIT=[0-9a-f]{40}")
        self.assertIn("HF_XET_HIGH_PERFORMANCE=auto", dockerfile)
        self.assertNotIn("HF_HUB_ENABLE_HF_TRANSFER", dockerfile)
        self.assertIn("--start-period=30m", dockerfile)

    def test_entrypoint_checks_runpod_territory_before_download(self) -> None:
        entrypoint = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        territory_check = entrypoint.index("check_deployment_territory")
        download = entrypoint.index('"${SCRIPT_DIR}/download_models.sh"')
        self.assertLess(territory_check, download)
        self.assertIn("RUNPOD_DC_ID", entrypoint)
        self.assertIn("MINIMAX_H3_SEPARATE_LICENSE", entrypoint)
        self.assertIn("US-*|EU-*", entrypoint)

    def test_required_minimax_notice_is_present(self) -> None:
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(
            "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement",
            notice,
        )

    def test_verifier_accepts_a_matching_file(self) -> None:
        payload = b"minimax-h3-test"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.bin").write_bytes(payload)
            manifest = {
                "files": [
                    {
                        "path": "model.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_models.py"),
                    "--manifest",
                    str(manifest_path),
                    "--root",
                    str(root),
                    "--mode",
                    "sha256",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified 1 files", result.stdout)

    def test_workflow_models_match_manifest_exactly(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_workflow.py"),
                "--workflow",
                str(WORKFLOW),
                "--manifest",
                str(MANIFEST),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 models", result.stdout)


if __name__ == "__main__":
    unittest.main()
