"""Execute the real entrypoint's pre-network branch in an isolated fake install."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash") if os.name != "nt" else "C:/Program Files/Git/bin/bash.exe"


@unittest.skipUnless(BASH and Path(BASH).is_file(), "bash is unavailable")
class CharacterEntrypointTests(unittest.TestCase):
    def make_install(self, temp):
        root = Path(temp) / "ComfyUI"
        director = root / "custom_nodes/ComfyUI_MiniMaxH3_Director"
        for relative, content in {
            "nodes/director.py": "# stub",
            "director/executor_core.py": "# MiniMAXH- local modification (2026)",
            "director/h3_motion_context.py": "# MiniMAXH- local fix for AIMixer issue #26",
        }.items():
            path = director / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        package = root / "custom_nodes/minimax_h3_ordered_storyboard"
        package.mkdir(parents=True)
        for name in ("__init__", "storyboard", "exporter", "mosaic_nodes", "turbo_nodes", "memory_nodes", "character_nodes", "refmod_nodes"):
            (package / f"{name}.py").write_text("# stub")
        (package / "web").mkdir()
        (package / "web/refmod_images.js").write_text("// stub")
        (package / "refmod_vendor").mkdir()
        for name in ("core.py", "bundle.py", "LICENSE"):
            (package / "refmod_vendor" / name).write_text("# stub")
        (package / "refmod_vendor/REVISION").write_text("f9462081e28794389b5a6c5067eb327412ad8ee7\n")
        fbc = root / "custom_nodes/ComfyUI-MiniMaxH3-FirstBlockCache"
        fbc.mkdir(parents=True)
        (fbc / "nodes.py").write_text("# stub")
        env = os.environ.copy()
        env.update({
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
            "COMFYUI_ROOT": root.as_posix(), "COMFYUI_MODEL_DIR": (root / "models").as_posix(),
            "MINIMAX_H3_DIRECTOR_ROOT": director.as_posix(), "MINIMAX_H3_FBC_ROOT": fbc.as_posix(),
            "MODEL_MANIFEST": (ROOT / "manifests/minimax_h3_i2v_upscale.json").as_posix(),
            "ACCEPT_MINIMAX_H3_LICENSE": "1", "MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY": "1",
            "MINIMAX_H3_SEPARATE_LICENSE": "0", "RUNPOD_DC_ID": "",
            "HF_TOKEN": "test-secret-not-logged", "CIVITAI_TOKEN": "test-secret-not-logged",
            "CIVITAI_API_TOKEN": "test-secret-not-logged", "H3_LORA_REQUIRED": "1",
            "H3_LORA_SELECTION": "all", "AUTO_MOSAIC_REQUIRED": "1",
            "MINIMAX_H3_ENTRYPOINT_SMOKE": "1", "MINIMAX_H3_RUNTIME_VARIANT": "community-cu128",
            "REQUIRE_COMFY_KITCHEN_CUDA": "0", "H3_CHARACTER_R2V": "1",
            "H3_PROFILE": "legacy", "H3_FAST_VAE": "1",
            "H3_R2V_MODEL": "official",
            "H3_R2V_COMPARE": "0",
            "H3_R2V_LORA_SELECTION": "none",
        })
        return root, env

    def run_entrypoint(self, env):
        result = subprocess.run(
            [BASH, str(ROOT / "scripts/entrypoint.sh")], env=env, cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        self.assertNotIn("test-secret-not-logged", result.stdout + result.stderr)
        return result

    def test_enabled_then_disabled_installs_four_then_original_three(self):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = self.make_install(temp)
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("entrypoint contract passed", result.stdout)
            workflows = root / "user/default/workflows"
            self.assertEqual(len(list(workflows.glob("*.json"))), 4)
            merged = json.loads((root / "user/default/minimax_h3_character_models.json").read_text())
            self.assertEqual(len(merged["files"]), 6)
            # Disabling opt-in on a later start restores the same original 3.
            env["H3_CHARACTER_R2V"] = "0"
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(workflows.glob("*.json"))), 3)
            for source, deployed in [
                ("minimax_h3_preset_01_quality.json", "01_MiniMax_H3_Quality_2x.json"),
                ("minimax_h3_preset_02_fast_fbcache.json", "02_MiniMax_H3_Fast_FBCache_2x.json"),
                ("minimax_h3_preset_03_turbo.json", "03_MiniMax_H3_Turbo_4_8step_768p_2x.json"),
            ]:
                self.assertEqual((ROOT / "workflows" / source).read_bytes(), (workflows / deployed).read_bytes())

    def test_enabled_fails_before_network_if_character_node_is_missing(self):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = self.make_install(temp)
            (root / "custom_nodes/minimax_h3_ordered_storyboard/character_nodes.py").unlink()
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
            self.assertNotIn("entrypoint contract passed", result.stdout)

    def test_r2v_only_ignores_legacy_models_and_restores_legacy_on_request(self):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = self.make_install(temp)
            self.assertEqual(self.run_entrypoint(env).returncode, 0)
            workflows = root / "user/default/workflows"
            personal = workflows / "My_MiniMax_H3_custom.json"
            personal.write_text("{}")
            env.update(H3_PROFILE="r2v", H3_TURBO_REQUIRED="1", H3_EXTRA_LORA_REQUIRED="1",
                       H3_EXTRA_LORA_LIST_URL="https://invalid.example/do-not-fetch",
                       COMFYUI_ARGS="--lowvram --vram-headroom 2")
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("R2VA-only entrypoint contract passed", result.stdout)
            self.assertIn("--fast fp16_accumulation", result.stdout)
            self.assertEqual(sorted(p.name for p in workflows.glob("*.json")),
                             ["04_MiniMax_H3_Character_R2V_2x.json", personal.name])
            self.assertEqual((workflows / "04_MiniMax_H3_Character_R2V_2x.json").read_bytes(),
                             (ROOT / "workflows/character_reveal_r2v_int8_2x.json").read_bytes())
            # Switching back restores shipped presets without removing personal files.
            env.update(H3_PROFILE="legacy", H3_CHARACTER_R2V="0")
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(workflows.glob("*.json"))), 4)
            self.assertTrue(personal.exists())
            self.assertFalse((workflows / "04_MiniMax_H3_Character_R2V_2x.json").exists())
            env.update(H3_PROFILE="r2v", H3_CHARACTER_R2V="ignored-in-r2v-profile")
            # Dedicated startup never requires Director or FirstBlockCache.
            shutil.rmtree(root / "custom_nodes/ComfyUI_MiniMaxH3_Director")
            shutil.rmtree(root / "custom_nodes/ComfyUI-MiniMaxH3-FirstBlockCache")
            self.assertEqual(self.run_entrypoint(env).returncode, 0)

    def test_r2v_fast_args_and_missing_nodes(self):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = self.make_install(temp)
            env.update(H3_PROFILE="r2v", COMFYUI_ARGS="--fast fp8_matrix_mult --vram-headroom 2")
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--fast fp16_accumulation fp8_matrix_mult --vram-headroom 2", result.stdout)
            env.update(H3_FAST_VAE="0", COMFYUI_ARGS="--vram-headroom 2")
            result = self.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("fp16_accumulation", result.stdout)
            (root / "custom_nodes/minimax_h3_ordered_storyboard/character_nodes.py").unlink()
            self.assertEqual(self.run_entrypoint(env).returncode, 77)

    def test_unknown_profile_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            _, env = self.make_install(temp)
            env["H3_PROFILE"] = "typo"
            self.assertEqual(self.run_entrypoint(env).returncode, 78)

    def test_r2v_missing_or_wrong_refmod_fails_before_download(self):
        for broken in ("refmod_nodes.py", "web/refmod_images.js", "refmod_vendor/core.py",
                       "refmod_vendor/bundle.py", "refmod_vendor/LICENSE", "wrong_revision"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
                root, env = self.make_install(temp)
                env.update(H3_PROFILE="r2v", H3_R2V_MODEL="dasiwa-v2")
                package = root / "custom_nodes/minimax_h3_ordered_storyboard"
                if broken == "wrong_revision":
                    (package / "refmod_vendor/REVISION").write_text("main")
                else:
                    (package / broken).unlink()
                result = self.run_entrypoint(env)
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
                self.assertNotIn("entrypoint contract passed", result.stdout)
                self.assertNotIn("installed one R2VA", result.stdout)


if __name__ == "__main__":
    unittest.main()
