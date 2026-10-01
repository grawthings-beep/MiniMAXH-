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
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_r2v_profile import prepare
from prepare_upscale_compare import build, merge_manifest, WORKFLOW
from verify_upscale_compare import verify


class ComparisonTests(unittest.TestCase):
    def test_profiles_keep_one_sampler_full_prompt_visible_lora_and_int8_references(self):
        for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
            baseline = prepare(profile)
            manifest, workflow = build(profile)
            verify(workflow, manifest)
            self.assertEqual(len(manifest["files"]), 7)
            self.assertEqual(manifest["total_bytes"], sum(f["size"] for f in manifest["files"]))
            self.assertEqual(prepare(profile), baseline)
            for kind in ("BasicScheduler", "KSamplerSelect", "UNETLoader", "MiniMaxH3R2VLoRA", "MiniMaxH3FullPrompt", "RandomNoise"):
                self.assertEqual(next(n for n in workflow["nodes"] if n["type"] == kind)["widgets_values"],
                                 next(n for n in baseline[2]["nodes"] if n["type"] == kind)["widgets_values"])
            for vae in ("int8", "x2-detail"):
                merged = merge_manifest(prepare(profile, vae_profile=vae)[0], manifest)
                self.assertEqual(len({f["path"] for f in merged["files"]}), len(merged["files"]))
                self.assertEqual(merged["total_bytes"], sum(f["size"] for f in merged["files"]))
                self.assertEqual(len([f for f in merged["files"] if f["path"].startswith("diffusion_models/")]), 1)

    def test_corrupt_wiring_models_and_layout_are_rejected(self):
        manifest, original = build("dasiwa-v2")
        for mutate in (
            lambda w: w["links"][0].__setitem__(3, 9999),
            lambda w: w["links"].append(copy.deepcopy(w["links"][0])),
            lambda w: w["nodes"][0]["outputs"][0]["links"].append(9999),
            lambda w: w["nodes"][0].__setitem__("pos", [-500, 0]),
            lambda w: next(n for n in w["nodes"] if n["type"] == "MiniMaxH3CompareUpscale")["properties"]["models"].pop(),
        ):
            workflow = copy.deepcopy(original)
            mutate(workflow)
            with self.assertRaises(RuntimeError):
                verify(workflow, manifest)
        bad = copy.deepcopy(manifest)
        bad["files"][-1]["sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            verify(original, bad)

    def test_asset_hashes_enforced_even_in_size_mode(self):
        from verify_models import verify_one
        config = json.loads((ROOT / "manifests/upscale_compare.json").read_text())
        self.assertEqual([f["size"] for f in config["files"]], [4461056, 31053198])
        self.assertEqual(config["files"][1]["license"], "CC-BY-NC-SA-4.0")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test"
            path.write_bytes(b"bad")
            for asset in config["files"]:
                self.assertTrue(asset["verify_sha256"])
                modified = {**asset, "path": "test", "size": 3}
                self.assertIn("sha256 mismatch", verify_one(Path(temp), modified, "size")[1])

    def test_node_import_is_lightweight_and_preflight_fails_closed(self):
        spec = importlib.util.spec_from_file_location("compare_unit", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/compare_nodes.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.NODE_CLASS_MAPPINGS["MiniMaxH3CompareUpscale"]
        self.assertTrue(cls.OUTPUT_NODE)
        self.assertEqual(cls.RETURN_TYPES, ("STRING",))
        self.assertTrue(cls.INPUT_TYPES()["required"]["mosaic_enabled"][1]["default"])
        from unittest.mock import patch
        from types import SimpleNamespace
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace(get_full_path=lambda *a: None)}):
            self.assertIn("missing", cls.VALIDATE_INPUTS())


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class ComparisonEntrypointTests(unittest.TestCase):
    def test_opt_in_all_profiles_and_restore_without_touching_personal_workflows(self):
        helper = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = helper.make_install(temp)
            env.update(H3_PROFILE="r2v", H3_R2V_COMPARE="1", H3_R2V_VAE="int8")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            package = root / "custom_nodes/minimax_h3_ordered_storyboard"
            (package / "compare_nodes.py").write_text("# stub")
            (package / "web/upscale_compare.js").write_text("// stub")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            decoder = root / "custom_nodes/minimax_h3_x2_vae"
            decoder.mkdir()
            for name in ("__init__.py", "vae_decode.py", "utils.py"):
                (decoder / name).write_text("# stub")
            (decoder / "REVISION").write_text("2e568dfe3e4f5e81da178bc845e05dcfbb64d55b\n")
            for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
                for vae in ("int8", "x2-detail"):
                    env.update(H3_R2V_MODEL=profile, H3_R2V_VAE=vae)
                    result = helper.run_entrypoint(env)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    deployed = root / "user/default/workflows"
                    self.assertEqual(len(list(deployed.glob("*.json"))), 2)
                    self.assertEqual(json.loads((deployed / WORKFLOW).read_text(encoding="utf-8")), build(profile)[1])
                    self.assertEqual(json.loads((deployed / "04_MiniMax_H3_Character_R2V_2x.json").read_text(encoding="utf-8")), prepare(profile, vae_profile=vae)[2])
                    hf = json.loads((root / "user/default/minimax_h3_r2v_hf_models.json").read_text())
                    self.assertFalse(any(f.get("auth") == "civitai" for f in hf["files"]))
            personal = deployed / "My comparison.json"
            personal.write_text("{}")
            env["H3_R2V_COMPARE"] = "0"
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((deployed / WORKFLOW).exists())
            self.assertTrue(personal.exists())
            manifest = json.loads((root / "user/default/minimax_h3_r2v_models.json").read_text())
            self.assertFalse(any("AnimeSharp" in f["path"] for f in manifest["files"]))
            env["H3_R2V_COMPARE"] = "typo"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)


if __name__ == "__main__":
    unittest.main()
