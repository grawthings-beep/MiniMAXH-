from __future__ import annotations

import hashlib
import importlib.util
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
        i2v_upscale = self.load_manifest("minimax_h3_i2v_upscale.json")
        r2v_upscale = self.load_manifest("minimax_h3_r2v_upscale.json")
        combined = self.load_manifest("minimax_h3_all.json")
        sets = {
            "i2v": {item["path"] for item in i2v["files"]},
            "r2v": {item["path"] for item in r2v["files"]},
            "i2v_upscale": {item["path"] for item in i2v_upscale["files"]},
            "r2v_upscale": {item["path"] for item in r2v_upscale["files"]},
            "combined": {item["path"] for item in combined["files"]},
        }
        upscaler = "upscale_models/RealESRGAN_x2plus.pth"
        self.assertEqual(len(sets["i2v"]), 4)
        self.assertEqual(len(sets["r2v"]), 4)
        self.assertEqual(len(sets["i2v_upscale"]), 5)
        self.assertEqual(len(sets["r2v_upscale"]), 5)
        self.assertEqual(len(sets["combined"]), 6)
        self.assertEqual(i2v_upscale["total_bytes"], 42_537_647_196)
        self.assertEqual(
            combined["total_bytes"] - i2v_upscale["total_bytes"],
            20_970_379_616,
        )
        self.assertTrue(any("fl2va" in path for path in sets["i2v"]))
        self.assertFalse(any("ref2va" in path for path in sets["i2v"]))
        self.assertTrue(any("ref2va" in path for path in sets["r2v"]))
        self.assertFalse(any("fl2va" in path for path in sets["r2v"]))
        self.assertEqual(sets["i2v_upscale"], sets["i2v"] | {upscaler})
        self.assertEqual(sets["r2v_upscale"], sets["r2v"] | {upscaler})
        self.assertEqual(sets["combined"], sets["i2v"] | sets["r2v"] | {upscaler})
        upscale_item = next(item for item in combined["files"] if item["path"] == upscaler)
        self.assertEqual(
            upscale_item["source_url"],
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.1/RealESRGAN_x2plus.pth",
        )
        self.assertEqual(upscale_item["size"], 67_061_725)
        self.assertEqual(
            upscale_item["sha256"],
            "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        )
        for manifest in (i2v, r2v, i2v_upscale, r2v_upscale, combined):
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
            "minimax_h3_i2v_upscale.json",
            "minimax_h3_i2v_hmmotion_lora_upscale.json",
            "minimax_h3_i2v_selectable_lora_upscale.json",
            "minimax_h3_i2v_easycache_upscale.json",
            "minimax_h3_r2v_upscale.json",
            "minimax_h3_r2v_easycache_upscale.json",
            "minimax_h3_i2v_auto_mosaic.json",
            "minimax_h3_i2v_easycache_auto_mosaic.json",
            "minimax_h3_i2v_upscale_auto_mosaic.json",
            "minimax_h3_i2v_easycache_upscale_auto_mosaic.json",
            "minimax_h3_i2v_hmmotion_lora_upscale_auto_mosaic.json",
            "minimax_h3_i2v_selectable_lora_upscale_auto_mosaic.json",
            "minimax_h3_r2v_auto_mosaic.json",
            "minimax_h3_r2v_easycache_auto_mosaic.json",
            "minimax_h3_r2v_upscale_auto_mosaic.json",
            "minimax_h3_r2v_easycache_upscale_auto_mosaic.json",
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
                "minimax_h3_i2v_upscale.json",
                "minimax_h3_i2v_upscale.json",
                "i2v",
                ["--expect-upscale"],
            ),
            (
                "minimax_h3_i2v_hmmotion_lora_upscale.json",
                "minimax_h3_i2v_upscale.json",
                "i2v",
                ["--expect-upscale", "--expect-lora"],
            ),
            (
                "minimax_h3_i2v_selectable_lora_upscale.json",
                "minimax_h3_i2v_upscale.json",
                "i2v",
                [
                    "--expect-upscale",
                    "--expect-lora",
                    "HMNSFW_AIO_V2.safetensors",
                    "--expect-lora-strength",
                    "0.5",
                ],
            ),
            (
                "minimax_h3_i2v_easycache_upscale.json",
                "minimax_h3_i2v_upscale.json",
                "i2v",
                ["--expect-easycache", "--expect-upscale"],
            ),
            (
                "minimax_h3_story_quality_lora_2x.json",
                "minimax_h3_i2v_upscale.json",
                "story",
                [
                    "--expect-upscale",
                    "--expect-lora",
                    "HMNSFW_AIO_V2.safetensors",
                    "--expect-lora-strength",
                    "0.5",
                ],
            ),
            (
                "minimax_h3_story_easycache_lora_2x.json",
                "minimax_h3_i2v_upscale.json",
                "story",
                [
                    "--expect-easycache",
                    "--expect-upscale",
                    "--expect-lora",
                    "HMNSFW_AIO_V2.safetensors",
                    "--expect-lora-strength",
                    "0.5",
                ],
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
            (
                "minimax_h3_r2v_upscale.json",
                "minimax_h3_r2v_upscale.json",
                "r2v",
                ["--expect-upscale", "--require-video-reference"],
            ),
            (
                "minimax_h3_r2v_easycache_upscale.json",
                "minimax_h3_r2v_upscale.json",
                "r2v",
                ["--expect-easycache", "--expect-upscale", "--require-video-reference"],
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

    def test_dockerfile_pins_comfyui_director_and_installs_story_workflows(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime@sha256:"
            "1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b",
            dockerfile,
        )
        self.assertIn("ARG COMFYUI_VERSION=v0.30.0", dockerfile)
        self.assertRegex(dockerfile, r"ARG COMFYUI_COMMIT=[0-9a-f]{40}")
        self.assertIn(
            "ARG MINIMAX_H3_DIRECTOR_COMMIT="
            "a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7",
            dockerfile,
        )
        self.assertIn("AIMixer/ComfyUI_MiniMaxH3_Director.git", dockerfile)
        self.assertIn("minimax_h3_director_segments_no_concat.patch", dockerfile)
        self.assertIn("minimax_h3_director_fl2v_motion_context.patch", dockerfile)
        self.assertIn("h3_motion_context.py'); p.write_bytes", dockerfile)
        self.assertIn("custom_nodes/minimax_h3_ordered_storyboard", dockerfile)
        self.assertIn("manifests/minimax_h3_i2v_upscale.json", dockerfile)
        self.assertIn("MODEL_VERIFY=size", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Quality.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Fast_EasyCache.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Quality.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Fast_EasyCache.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Quality_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Quality_Selectable_LoRA_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_I2V_Fast_EasyCache_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Quality_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_R2V_Fast_EasyCache_2x.json", dockerfile)
        self.assertIn("MiniMax_H3_Story_Quality_Selectable_LoRA_2x.json", dockerfile)
        self.assertIn(
            "MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x.json",
            dockerfile,
        )
        self.assertIn("HF_XET_HIGH_PERFORMANCE=auto", dockerfile)
        self.assertNotIn("HF_HUB_ENABLE_HF_TRANSFER", dockerfile)
        self.assertNotIn("ComfyUI-INT8-Fast", dockerfile)
        self.assertNotIn("ComfyUI_sol-attn_Blackwell", dockerfile)
        self.assertIn("PIP_BREAK_SYSTEM_PACKAGES=1", dockerfile)
        self.assertIn("REQUIRE_COMFY_KITCHEN_CUDA=1", dockerfile)
        self.assertIn("--start-period=30m", dockerfile)
        self.assertIn("H3_LORA_REQUIRED=1", dockerfile)
        self.assertIn("H3_LORA_SELECTION=all", dockerfile)
        self.assertIn("H3_LORA_REPO_ID=uwgm/nikke-civitai-backup", dockerfile)
        self.assertIn("civitai.red/api/download/models/3206518?fileId=3088013", dockerfile)
        self.assertIn("ultralytics.__version__ == '8.4.104'", dockerfile)
        self.assertIn("AUTO_MOSAIC_REQUIRED=1", dockerfile)
        self.assertIn("verify_auto_mosaic_workflows.py", dockerfile)
        self.assertIn("'WanAutoMosaicVideo' in p.NODE_CLASS_MAPPINGS", dockerfile)
        self.assertIn("MINIMAX_H3_ENTRYPOINT_SMOKE=1", dockerfile)

        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("minimax_h3_director_fl2v_motion_context.patch", ci)
        self.assertIn("apply --recount", ci)
        self.assertIn('replace(b"\\r\\n", b"\\n")', ci)

    def test_downloader_supports_pinned_external_upscaler(self) -> None:
        downloader = (ROOT / "scripts" / "download_models.sh").read_text(encoding="utf-8")
        self.assertIn("manifests/minimax_h3_i2v_upscale.json", downloader)
        self.assertIn('VERIFY_MODE="${MODEL_VERIFY:-size}"', downloader)
        self.assertIn('if "source_url" in item', downloader)
        self.assertIn("EXTERNAL_RECORDS", downloader)
        self.assertIn("download_external_files", downloader)
        self.assertIn("complete manifest fallback", downloader)

    def test_entrypoint_checks_licensee_territory_before_download(self) -> None:
        entrypoint = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
        territory_check = entrypoint.index("check_licensee_territory")
        download = entrypoint.index('"${SCRIPT_DIR}/download_models.sh"')
        self.assertLess(territory_check, download)
        self.assertIn("RUNPOD_DC_ID", entrypoint)
        self.assertIn("MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY", entrypoint)
        self.assertIn("MINIMAX_H3_SEPARATE_LICENSE", entrypoint)
        self.assertIn("US-*|EU-*", entrypoint)
        self.assertIn("WARNING: RunPod data center", entrypoint)
        self.assertNotIn("MINIMAX_H3_DEPLOYMENT_ALLOWED", entrypoint)
        self.assertNotIn("Choose an eligible data center", entrypoint)
        self.assertIn("manifests/minimax_h3_i2v_upscale.json", entrypoint)
        self.assertIn('any("ref2va" in item["path"]', entrypoint)
        self.assertIn("I2V-only manifest selected", entrypoint)
        self.assertIn("--require-comfy-kitchen-cuda", entrypoint)
        self.assertIn("--fast-disk can make H3 model offload much slower", entrypoint)
        self.assertIn("download_lora.py", entrypoint)
        self.assertIn("download_civitai_lora.py", entrypoint)
        self.assertIn("HMMOTION_DOWNLOAD_PID", entrypoint)
        self.assertIn("CIVITAI_DOWNLOAD_PID", entrypoint)
        self.assertIn("AUTO_MOSAIC_DOWNLOAD_PID", entrypoint)
        self.assertIn("CIVITAI_API_TOKEN", entrypoint)
        self.assertIn("download_auto_mosaic.py", entrypoint)
        self.assertIn("entrypoint contract passed before network/model startup", entrypoint)
        self.assertIn("MODEL_DOWNLOAD_PID", entrypoint)
        self.assertIn("MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x.json", entrypoint)
        self.assertIn("MiniMax_H3_I2V_Quality_Selectable_LoRA_2x.json", entrypoint)
        self.assertIn("MiniMax_H3_Story_Quality_Selectable_LoRA_2x.json", entrypoint)
        self.assertIn(
            "MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x.json",
            entrypoint,
        )
        self.assertIn("required Director segments-mode memory patch is missing", entrypoint)
        self.assertIn("required Director FL2V Motion Context fix is missing", entrypoint)
        self.assertIn('${MODEL_DIR}/loras/HMNSFW_AIO_V2.safetensors', entrypoint)

    def test_runpod_template_uses_safe_performance_defaults(self) -> None:
        template = json.loads((ROOT / "runpod-template.example.json").read_text(encoding="utf-8"))
        self.assertEqual(template["imageName"], "ghcr.io/grawthings-beep/minimax-h3-i2v:0.9.0")
        self.assertEqual(template["env"]["MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY"], "0")
        self.assertNotIn("MINIMAX_H3_DEPLOYMENT_ALLOWED", template["env"])
        self.assertEqual(template["env"]["REQUIRE_COMFY_KITCHEN_CUDA"], "1")
        self.assertEqual(template["env"]["COMFYUI_ARGS"], "--vram-headroom 2")
        self.assertNotIn("--fast-disk", template["env"]["COMFYUI_ARGS"])
        self.assertEqual(template["env"]["HF_TOKEN"], "")
        self.assertEqual(template["env"]["CIVITAI_TOKEN"], "")
        self.assertEqual(template["env"]["CIVITAI_API_TOKEN"], "")
        self.assertEqual(template["env"]["AUTO_MOSAIC_REQUIRED"], "1")
        self.assertEqual(template["env"]["H3_LORA_REQUIRED"], "1")
        self.assertEqual(template["env"]["H3_LORA_SELECTION"], "all")
        self.assertEqual(template["env"]["H3_LORA_REPO_ID"], "uwgm/nikke-civitai-backup")
        self.assertEqual(
            template["env"]["MODEL_MANIFEST"],
            "/opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json",
        )
        self.assertEqual(template["env"]["MODEL_VERIFY"], "size")
        self.assertEqual(
            template["env"]["H3_LORA_SOURCE_PATH"],
            "hmmotion_minimax-h3_epoch12.safetensors",
        )

    def test_required_minimax_notice_is_present(self) -> None:
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(
            "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement",
            notice,
        )
        self.assertIn("Real-ESRGAN_x2plus", notice)
        self.assertIn("BSD 3-Clause License", notice)
        self.assertIn("hmmotion_minimax-h3_epoch12.safetensors", notice)
        self.assertIn("HMNSFW_AIO_V2.safetensors", notice)
        self.assertIn("ComfyUI_MiniMaxH3_Director", notice)
        self.assertIn("a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7", notice)

    def test_director_memory_patch_is_narrow_and_prominently_marked(self) -> None:
        patch = (
            ROOT / "patches" / "minimax_h3_director_segments_no_concat.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("director/executor_core.py", patch)
        self.assertIn("MiniMAXH- local modification (2026)", patch)
        self.assertIn('plan.export_mode == "segments"', patch)
        self.assertNotIn("h3_motion_context.py", patch)

    def test_director_motion_context_patch_only_retimes_fl2v_last_frame(self) -> None:
        patch = (
            ROOT / "patches" / "minimax_h3_director_fl2v_motion_context.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("director/h3_motion_context.py", patch)
        self.assertIn("MiniMAXH- local fix for AIMixer issue #26", patch)
        self.assertIn("kf[CTX_FRAME_KEY] = resolved", patch)
        self.assertNotIn("executor_core.py", patch)

    def test_lora_downloader_validates_safetensors_without_exposing_token(self) -> None:
        script_path = ROOT / "scripts" / "download_lora.py"
        source = script_path.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("HF_TOKEN"', source)
        self.assertNotIn("print(token", source)
        self.assertIn("resolved_revision", source)
        self.assertIn("inspect_safetensors", source)
        self.assertIn("H3_LORA_SHA256", source)

        spec = importlib.util.spec_from_file_location("download_lora", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        header = json.dumps(
            {"lora.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "test.safetensors"
            model.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
            self.assertEqual(module.inspect_safetensors(model), (model.stat().st_size, 1))

            scripts_path = str(ROOT / "scripts")
            sys.path.insert(0, scripts_path)
            sys.modules["download_lora"] = module
            try:
                civitai_path = ROOT / "scripts" / "download_civitai_lora.py"
                civitai_spec = importlib.util.spec_from_file_location(
                    "download_civitai_lora", civitai_path
                )
                self.assertIsNotNone(civitai_spec)
                self.assertIsNotNone(civitai_spec.loader)
                civitai = importlib.util.module_from_spec(civitai_spec)
                civitai_spec.loader.exec_module(civitai)
            finally:
                sys.path.remove(scripts_path)
            self.assertEqual(civitai.EXPECTED_SIZE, 310_168_344)
            self.assertEqual(
                civitai.EXPECTED_SHA256,
                "608e4212f2788b6063330ff1196fc1f4b4228cfd9a413a63c198a09d7e4a61cb",
            )
            self.assertEqual(
                civitai.validate_source_url(civitai.DEFAULT_SOURCE_URL),
                civitai.DEFAULT_SOURCE_URL,
            )
            with self.assertRaises(ValueError):
                civitai.validate_source_url(
                    civitai.DEFAULT_SOURCE_URL + "&token=test-secret"
                )
            request = civitai.build_civitai_request(
                civitai.DEFAULT_SOURCE_URL, "test-secret", 1024
            )
            self.assertNotIn("test-secret", request.full_url)
            self.assertEqual(
                request.unredirected_hdrs["Authorization"], "Bearer test-secret"
            )
            self.assertEqual(request.headers["Range"], "bytes=1024-")
            redirected = civitai.urllib.request.HTTPRedirectHandler().redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://signed-storage.example/model.safetensors",
            )
            self.assertIsNotNone(redirected)
            self.assertIsNone(redirected.get_header("Authorization"))
            self.assertEqual(redirected.get_header("Range"), "bytes=1024-")
            size, tensors = civitai.verify_download(
                model, model.stat().st_size, hashlib.sha256(model.read_bytes()).hexdigest()
            )
            self.assertEqual((size, tensors), (model.stat().st_size, 1))

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
