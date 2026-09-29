"""R2VA-only model contract and real shell orchestration with network stubs."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import test_character_entrypoint as entrypoint_tests
import test_character_r2v as character_tests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/minimax_h3_r2v_int8_upscale.json"


class R2VOnlyWorkflowTests(character_tests.CharacterTests):
    def setUp(self):
        self.workflow = json.loads((ROOT / "workflows/character_reveal_r2v_int8_2x.json").read_text(encoding="utf-8"))

    def test_generated_workflow_and_existing_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT / "scripts/build_character_workflow.py"),
                            "--int8-vae", "--output-dir", directory], check=True, capture_output=True)
            self.assertEqual(json.loads((Path(directory) / "character_reveal_r2v_int8_2x.json").read_text(encoding="utf-8")), self.workflow)
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/verify_workflow.py"),
            "--workflow", str(ROOT / "workflows/character_reveal_r2v_int8_2x.json"),
            "--manifest", str(MANIFEST), "--mode", "r2v", "--expect-upscale",
            "--expect-memory-safe-decode", "--expect-auto-mosaic",
            "--auto-mosaic-manifest", str(ROOT / "manifests/auto_mosaic.json"),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_int8_assets_pinned_without_unused_models(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["total_bytes"], sum(f["size"] for f in manifest["files"]))
        self.assertEqual(manifest["total_bytes"], 40_140_903_884)
        self.assertEqual(manifest["revision"], "bf92c4091e333e69b8ca1998e0a669f15cb0832b")
        self.assertEqual(len(manifest["files"]), 5)
        self.assertFalse(any("fl2va" in f["path"] or "loras/" in f["path"] for f in manifest["files"]))
        vae = next(f for f in manifest["files"] if "video_vae" in f["path"])
        self.assertEqual(vae["size"], 2_811_065_184)
        self.assertEqual(vae["sha256"], "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e")
        legacy = json.loads((ROOT / "manifests/minimax_h3_r2v_upscale.json").read_text())
        old = {f["path"]: f for f in legacy["files"]}
        for asset in manifest["files"]:
            if asset["path"] in old:
                self.assertEqual(asset, old[asset["path"]])
        selected = next(n for n in self.workflow["nodes"] if n["id"] == 119)
        self.assertEqual(selected["widgets_values"], [Path(vae["path"]).name])

    def test_dedicated_template_has_no_legacy_download_configuration(self):
        template = json.loads((ROOT / "runpod-template.r2v-cu130.example.json").read_text())
        self.assertEqual(template["env"]["H3_PROFILE"], "r2v")
        self.assertIn("--fast fp16_accumulation", template["env"]["COMFYUI_ARGS"])
        self.assertFalse(any("LORA" in k or "TURBO" in k for k in template["env"]))
        self.assertEqual(template["volumeInGb"], 0)


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class R2VOnlyOrchestrationTests(unittest.TestCase):
    def exercise(self, *, download_failure=False, verify_failure=False, checkpoint_failure=False, profile="official"):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = entrypoint_tests.CharacterEntrypointTests().make_install(temp)
            project = Path(temp) / "project"
            scripts = project / "scripts"
            scripts.mkdir(parents=True)
            for name in ("entrypoint.sh", "entrypoint_r2v.sh", "prepare_r2v_profile.py", "verify_workflow.py"):
                shutil.copyfile(ROOT / "scripts" / name, scripts / name)
            shutil.copytree(ROOT / "manifests", project / "manifests")
            (project / "workflows").mkdir()
            shutil.copyfile(ROOT / "workflows/character_reveal_r2v_int8_2x.json",
                            project / "workflows/character_reveal_r2v_int8_2x.json")
            (scripts / "preflight.py").write_text(
                "import os,json\nm=json.load(open(os.environ['MODEL_MANIFEST']))\n"
                "assert len(m['files'])==5\nassert not any('fl2va' in f['path'] for f in m['files'])\n"
                "print('PREFLIGHT_R2V_ONLY')\n")
            (scripts / "download_models.sh").write_text(
                '#!/usr/bin/env bash\nset -eu\necho BASE_DOWNLOAD_ONLY\n'
                'python -c "import os,json; m=json.load(open(os.environ[\'MODEL_MANIFEST\'])); '
                'assert not any(f.get(\'auth\') for f in m[\'files\']); '
                'assert len(m[\'files\']) == int(os.environ[\'TEST_COMPANION_COUNT\'])"\n'
                'exit "${TEST_DOWNLOAD_FAILURE:-0}"\n', newline="\n")
            (scripts / "download_models.sh").chmod(0o755)
            (scripts / "download_auto_mosaic.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "p=Path(os.environ['COMFYUI_MODEL_DIR'])/'auto_mosaic/ntd11_anime_nsfw_segm_v5.pt'\n"
                "p.write_bytes(b'stub')\nprint('MOSAIC_DOWNLOAD_ONLY')\n")
            (scripts / "verify_models.py").write_text(
                "import os,sys\nprint('VERIFY_BEFORE_LAUNCH')\n"
                "sys.exit(int(os.environ.get('TEST_VERIFY_FAILURE','0')))\n")
            (scripts / "download_civitai_models.py").write_text(
                "import os,sys\nprint('CHECKPOINT_DOWNLOAD_SELECTED')\n"
                "sys.exit(int(os.environ.get('TEST_CHECKPOINT_FAILURE','0')))\n")
            for name in ("download_lora.py", "download_civitai_lora.py", "download_turbo_lora.py", "download_extra_loras.py"):
                (scripts / name).write_text("raise RuntimeError('UNEXPECTED_LEGACY_DOWNLOAD')\n")
            (root / "main.py").write_text("import sys\nprint('COMFY_LAUNCHED',sys.argv[1:])\n")
            env.update(H3_PROFILE="r2v", MINIMAX_H3_ENTRYPOINT_SMOKE="0",
                       H3_R2V_MODEL=profile, TEST_COMPANION_COUNT="5" if profile == "official" else "4",
                       H3_TURBO_REQUIRED="1", H3_EXTRA_LORA_REQUIRED="1",
                       COMFYUI_ARGS="--lowvram --vram-headroom 2",
                       TEST_DOWNLOAD_FAILURE="9" if download_failure else "0",
                       TEST_VERIFY_FAILURE="10" if verify_failure else "0",
                       TEST_CHECKPOINT_FAILURE="11" if checkpoint_failure else "0")
            result = subprocess.run([entrypoint_tests.BASH, str(scripts / "entrypoint.sh")],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotIn("UNEXPECTED_LEGACY_DOWNLOAD", result.stdout + result.stderr)
            self.assertNotIn("test-secret-not-logged", result.stdout + result.stderr)
            return result

    def test_selected_downloads_then_verification_then_launch(self):
        result = self.exercise()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BASE_DOWNLOAD_ONLY", result.stdout)
        self.assertIn("MOSAIC_DOWNLOAD_ONLY", result.stdout)
        self.assertLess(result.stdout.index("VERIFY_BEFORE_LAUNCH"), result.stdout.index("COMFY_LAUNCHED"))
        self.assertIn("'--fast', 'fp16_accumulation'", result.stdout)

    def test_download_failure_does_not_launch_comfy(self):
        result = self.exercise(download_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("COMFY_LAUNCHED", result.stdout)

    def test_verification_failure_does_not_launch_comfy(self):
        result = self.exercise(verify_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("COMFY_LAUNCHED", result.stdout)

    def test_dasiwa_companions_exclude_official_checkpoint_and_wait_for_selected_checkpoint(self):
        for profile in ("dasiwa-v2", "dasiwa-turbo-v2"):
            result = self.exercise(profile=profile)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertLess(result.stdout.index("CHECKPOINT_DOWNLOAD_SELECTED"), result.stdout.index("COMFY_LAUNCHED"))
            result = self.exercise(profile=profile, checkpoint_failure=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("COMFY_LAUNCHED", result.stdout)
