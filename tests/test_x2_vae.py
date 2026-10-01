"""X2 profile boundaries, executable validation, corruption and startup failures."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_character_entrypoint as entrypoint_tests

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_r2v_profile").prepare
verifier = load("verify_workflow")


class X2ProfileTests(unittest.TestCase):
    def test_all_model_and_vae_combinations_validate(self):
        for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
            baseline = prepare(profile)
            manifest, companions, workflow = prepare(profile, vae_profile="x2-detail")
            self.assertEqual(len(manifest["files"]), 4)
            self.assertEqual(manifest["total_bytes"], sum(f["size"] for f in manifest["files"]))
            self.assertFalse(any(f.get("auth") == "civitai" for f in companions["files"]))
            self.assertTrue(any(f.get("verify_sha256") for f in companions["files"]))
            self.assertEqual([f for f in manifest["files"] if f["path"].startswith("diffusion_models/")],
                             [f for f in baseline[0]["files"] if f["path"].startswith("diffusion_models/")])
            for kind in ("KSamplerSelect", "BasicScheduler", "RandomNoise", "ResolutionSelector", "MiniMaxH3FullPrompt"):
                before = next(n for n in baseline[2]["nodes"] if n["type"] == kind)
                after = next(n for n in workflow["nodes"] if n["type"] == kind)
                self.assertEqual(before["widgets_values"], after["widgets_values"])
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                for vae, data in (("int8", baseline), ("x2-detail", (manifest, companions, workflow))):
                    model_path, workflow_path = Path(directory) / "manifest.json", Path(directory) / "workflow.json"
                    model_path.write_text(json.dumps(data[0]), encoding="utf-8")
                    workflow_path.write_text(json.dumps(data[2]), encoding="utf-8")
                    result = subprocess.run([
                        sys.executable, str(ROOT / "scripts/verify_workflow.py"),
                        "--workflow", str(workflow_path), "--manifest", str(model_path), "--mode", "r2v",
                        "--expect-x2-vae" if vae == "x2-detail" else "--expect-upscale",
                        "--expect-memory-safe-decode", "--expect-auto-mosaic",
                        "--auto-mosaic-manifest", str(ROOT / "manifests/auto_mosaic.json"),
                    ], capture_output=True, text=True, encoding="utf-8")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(prepare(profile), baseline, "X2 selection must not mutate the stock preset")

    def test_corrupt_wiring_and_double_upscale_are_rejected(self):
        manifest, _, workflow = prepare("official", vae_profile="x2-detail")
        invalid = copy.deepcopy(workflow)
        invalid["nodes"].append({"id": 999, "type": "ImageUpscaleWithModel"})
        with self.assertRaisesRegex(RuntimeError, "another 2x"):
            verifier.verify_x2_vae(invalid, manifest)
        invalid = copy.deepcopy(workflow)
        invalid["links"][0][3] = 9999
        with self.assertRaises(RuntimeError):
            verifier.verify_x2_vae(invalid, manifest)
        invalid = copy.deepcopy(workflow)
        guard = next(n for n in invalid["nodes"] if n["type"] == "MiniMaxH3ReleaseVRAMLatent")
        invalid["links"] = [l for l in invalid["links"] if l[1] != guard["id"]]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            verifier.verify_memory_safe_decode(invalid, x2=True)

    def test_x2_hash_is_checked_even_in_size_mode(self):
        verify = load("verify_models").verify_one
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "vae.bin").write_bytes(b"wrong")
            asset = {"path": "vae.bin", "size": 5, "sha256": "0" * 64, "verify_sha256": True}
            self.assertIn("sha256 mismatch", verify(root, asset, "size")[1])
            asset.pop("verify_sha256")
            self.assertEqual(verify(root, asset, "size")[1], "ok")

    def test_invalid_profile_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "H3_R2V_VAE"):
            prepare("official", vae_profile="x2-typo")


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class X2EntrypointTests(unittest.TestCase):
    def test_pre_network_install_restore_and_missing_decoder(self):
        helper = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = helper.make_install(temp)
            env.update(H3_PROFILE="r2v", H3_R2V_VAE="x2-detail")
            result = helper.run_entrypoint(env)
            self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
            decoder = root / "custom_nodes/minimax_h3_x2_vae"
            decoder.mkdir()
            for name in ("__init__.py", "vae_decode.py", "utils.py"):
                (decoder / name).write_text("# synthetic source\n")
            (decoder / "REVISION").write_text("wrong\n")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            (decoder / "REVISION").write_text("2e568dfe3e4f5e81da178bc845e05dcfbb64d55b\n")
            for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
                env["H3_R2V_MODEL"] = profile
                result = helper.run_entrypoint(env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("VAE=x2-detail", result.stdout)
                deployed = root / "user/default/workflows/04_MiniMax_H3_Character_R2V_2x.json"
                workflow = json.loads(deployed.read_text(encoding="utf-8"))
                self.assertIn("MiniMaxH3VAEDecodeFast", {n["type"] for n in workflow["nodes"]})
            env.update(H3_R2V_MODEL="official", H3_R2V_VAE="int8")
            result = helper.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(deployed.read_bytes(), (ROOT / "workflows/character_reveal_r2v_int8_2x.json").read_bytes())
            env["H3_R2V_VAE"] = "typo"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)


if __name__ == "__main__":
    unittest.main()
