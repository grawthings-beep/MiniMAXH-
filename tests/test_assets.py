from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
WORKFLOWS = ROOT / "workflows"


class AssetTests(unittest.TestCase):
    def load_manifest(self, name: str) -> dict[str, object]:
        return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))

    def test_manifests_cover_i2v_r2v_and_combined_download(self) -> None:
        i2v = self.load_manifest("minimax_h3_i2v.json")
        r2v = self.load_manifest("minimax_h3_r2v.json")
        combined = self.load_manifest("minimax_h3_all.json")
        sets = {
            "i2v": {item["path"] for item in i2v["files"]},
            "r2v": {item["path"] for item in r2v["files"]},
            "combined": {item["path"] for item in combined["files"]},
        }
        self.assertEqual(len(sets["i2v"]), 4)
        self.assertEqual(len(sets["r2v"]), 4)
        self.assertEqual(len(sets["combined"]), 5)
        self.assertTrue(any("fl2va" in path for path in sets["i2v"]))
        self.assertFalse(any("ref2va" in path for path in sets["i2v"]))
        self.assertTrue(any("ref2va" in path for path in sets["r2v"]))
        self.assertFalse(any("fl2va" in path for path in sets["r2v"]))
        self.assertEqual(sets["combined"], sets["i2v"] | sets["r2v"])
        for manifest in (i2v, r2v, combined):
            self.assertEqual(
                manifest["total_bytes"], sum(item["size"] for item in manifest["files"])
            )
            for item in manifest["files"]:
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(item["size"], 0)

    def test_upstream_workflows_are_pinned(self) -> None:
        expected = {
            "minimax_h3_i2v.json": "bb71aecdd3c0b62e56eafe03acb14d1cfeabec7072eaed9cbdf473c2aaf73009",
            "upstream_minimax_h3_r2v.json": "6d36bacc5e09ae9168e703cf42d817ee78d44329b96369a3fdce123750e98247",
        }
        for name, digest in expected.items():
            canonical = (WORKFLOWS / name).read_text(encoding="utf-8").encode("utf-8")
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)

    def test_generated_workflows_are_reproducible(self) -> None:
        generated_names = (
            "minimax_h3_r2v.json",
            "minimax_h3_i2v_easycache.json",
            "minimax_h3_r2v_easycache.json",
        )
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_workflows.py"),
                    "--i2v-source",
                    str(WORKFLOWS / "minimax_h3_i2v.json"),
                    "--r2v-upstream",
                    str(WORKFLOWS / "upstream_minimax_h3_r2v.json"),
                    "--output-dir",
                    temp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in generated_names:
                self.assertEqual((Path(temp) / name).read_bytes(), (WORKFLOWS / name).read_bytes())

    def test_quality_and_fast_workflows_verify(self) -> None:
        cases = (
            ("minimax_h3_i2v.json", "minimax_h3_i2v.json", "i2v", []),
            (
                "minimax_h3_i2v_easycache.json",
                "minimax_h3_i2v.json",
                "i2v",
                ["--expect-easycache"],
            ),
            (
                "minimax_h3_r2v.json",
                "minimax_h3_r2v.json",
                "r2v",
                ["--require-video-reference"],
            ),
            (
                "minimax_h3_r2v_easycache.json",
                "minimax_h3_r2v.json",
                "r2v",
                ["--expect-easycache", "--require-video-reference"],
            ),
        )
        for workflow, manifest, mode, extra in cases:
            with self.subTest(workflow=workflow):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "verify_workflow.py"),
                        "--workflow",
                        str(WORKFLOWS / workflow),
                        "--manifest",
                        str(MANIFESTS / manifest),
                        "--mode",
                        mode,
                        *extra,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(mode.upper(), result.stdout)

    def test_dockerfile_pins_comfyui_and_installs_four_workflows(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:"
            "1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b",
            dockerfile,
        )
        self.assertIn("ARG COMFYUI_VERSION=v0.30.0", dockerfile)
        self.assertRegex(dockerfile, r"ARG COMFYUI_COMMIT=[0-9a-f]{40}")
        self.assertIn("manifests/minimax_h3_all.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Quality.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Fast_EasyCache.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Quality.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Fast_EasyCache.json", dockerfile)
        self.assertIn("HF_XET_HIGH_PERFORMANCE=auto", dockerfile)
        self.assertNotIn("HF_HUB_ENABLE_HF_TRANSFER", dockerfile)
        self.assertNotIn("ComfyUI-INT8-Fast", dockerfile)
        self.assertNotIn("ComfyUI_sol-attn_Blackwell", dockerfile)
        self.assertIn("REQUIRE_COMFY_KITCHEN_CUDA=1", dockerfile)
        self.assertIn("--start-period=30m", dockerfile)

    def test_entrypoint_checks_runpod_territory_before_download(self) -> None:
        entrypoint = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        territory_check = entrypoint.index("check_deployment_territory")
        download = entrypoint.index('"${SCRIPT_DIR}/download_models.sh"')
        self.assertLess(territory_check, download)
        self.assertIn("RUNPOD_DC_ID", entrypoint)
        self.assertIn("MINIMAX_H3_SEPARATE_LICENSE", entrypoint)
        self.assertIn("US-*|EU-*", entrypoint)
        self.assertIn("manifests/minimax_h3_all.json", entrypoint)
        self.assertIn("--require-comfy-kitchen-cuda", entrypoint)
        self.assertIn("--fast-disk can make H3 model offload much slower", entrypoint)

    def test_runpod_template_uses_safe_performance_defaults(self) -> None:
        template = json.loads((ROOT / "runpod-template.example.json").read_text(encoding="utf-8"))
        self.assertEqual(template["imageName"], "ghcr.io/grawthings-beep/minimax-h3-i2v:0.3.0")
        self.assertEqual(template["env"]["REQUIRE_COMFY_KITCHEN_CUDA"], "1")
        self.assertEqual(template["env"]["COMFYUI_ARGS"], "--vram-headroom 2")
        self.assertNotIn("--fast-disk", template["env"]["COMFYUI_ARGS"])

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


if __name__ == "__main__":
    unittest.main()
