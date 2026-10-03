import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import test_character_entrypoint as entrypoint_tests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_character_swap import build, WORKFLOW, LORA
from prepare_r2v_profile import prepare
from verify_character_swap import verify


class CharacterSwapTests(unittest.TestCase):
    def test_all_supported_profiles_layout_routes_and_selected_model(self):
        for p in ("official", "dasiwa-v2"):
            for v in ("int8", "x2-detail"):
                baseline = prepare(p, vae_profile=v)
                m, w = build(p, vae_profile=v)
                verify(w, m)
                self.assertEqual(prepare(p, vae_profile=v), baseline)
                self.assertEqual(len([f for f in m["files"] if f["path"].startswith("diffusion_models/")]), 1)
                self.assertEqual(m["total_bytes"], sum(f["size"] for f in m["files"]))
                self.assertEqual(len([n for n in w["nodes"] if n["type"] == "WanAutoMosaicVideo"]), 1)
                self.assertEqual(len([n for n in w["nodes"] if n["type"] == "MiniMaxH3CreateCharacterRefMod"]), 0)
                self.assertEqual(next(n for n in w["nodes"] if n["type"] == "MiniMaxH3CharacterSwap")["inputs"][1]["type"], "H3_CHARACTER_IMAGES")
                self.assertFalse(w["extra"]["character_r2v"]["quality_tested_on_gpu"])
        with self.assertRaises(ValueError):
            build("dasiwa-turbo-v2")

    def test_corrupted_workflow_rejected(self):
        m, original = build("dasiwa-v2", vae_profile="x2-detail")
        mutations = [
            lambda w: w["links"][0].__setitem__(3, 9999),
            lambda w: w["links"].append(copy.deepcopy(w["links"][0])),
            lambda w: w["nodes"][0]["outputs"][0]["links"].append(9999),
            lambda w: w["nodes"][0].__setitem__("pos", [-100, 0]),
            lambda w: next(n for n in w["nodes"] if n["type"] == "MiniMaxH3R2VLoRA")["widgets_values"].__setitem__(2, False),
            lambda w: next(n for n in w["nodes"] if n["type"] == "MiniMaxH3SwapAudio")["widgets_values"].__setitem__(0, "generated"),
            lambda w: next(n for n in w["nodes"] if n["type"] == "MiniMaxH3R2VLoRA")["properties"]["models"].clear(),
        ]
        for mutate in mutations:
            w = copy.deepcopy(original)
            mutate(w)
            with self.assertRaises(RuntimeError):
                verify(w, m)

    def test_pinned_asset_sha_is_mandatory_even_size_mode(self):
        from verify_models import verify_one
        asset = json.loads((ROOT / "manifests/character_swap.json").read_text())["files"][0]
        self.assertEqual(asset["path"], f"loras/{LORA}")
        self.assertEqual(asset["size"], 155110320)
        self.assertTrue(asset["verify_sha256"])
        self.assertIn("62407e0cc8089c363abd9ce4b0b27662abb237af", asset["source_url"])
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "fixture").write_bytes(b"bad")
            self.assertIn("sha256 mismatch", verify_one(Path(temp), {**asset, "path": "fixture", "size": 3}, "size")[1])

    def test_lightweight_import_and_canvas_limits(self):
        spec = importlib.util.spec_from_file_location("swap_unit", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/swap_nodes.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(len(module.NODE_CLASS_MAPPINGS), 4)
        for cls in module.NODE_CLASS_MAPPINGS.values():
            self.assertIn("required", cls.INPUT_TYPES())
        self.assertEqual(module.canvas(1920, 1080, .4), (832, 448))
        self.assertEqual(module.canvas(1920, 1080, .4, rotation=1), (448, 832))
        for mp in (float("nan"), float("inf"), 2, -.4):
            with self.assertRaises(ValueError):
                module.canvas(1920, 1080, mp)


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class CharacterSwapEntrypointTests(unittest.TestCase):
    def test_opt_in_combinations_missing_nodes_restore_and_secret_redaction(self):
        helper = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = helper.make_install(temp)
            env.update(H3_PROFILE="r2v", H3_CHARACTER_SWAP="1", H3_R2V_VAE="int8")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            package = root / "custom_nodes/minimax_h3_ordered_storyboard"
            for name in ("swap_nodes.py", "compare_nodes.py", "web/upscale_compare.js"):
                (package / name).write_text("# stub")
            decoder = root / "custom_nodes/minimax_h3_x2_vae"
            decoder.mkdir()
            for name in ("__init__.py", "vae_decode.py", "utils.py"):
                (decoder / name).write_text("# stub")
            (decoder / "REVISION").write_text("2e568dfe3e4f5e81da178bc845e05dcfbb64d55b\n")
            for profile in ("official", "dasiwa-v2"):
                for vae in ("int8", "x2-detail"):
                    for compare in ("0", "1"):
                        env.update(H3_R2V_MODEL=profile, H3_R2V_VAE=vae, H3_R2V_COMPARE=compare)
                        result = helper.run_entrypoint(env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        deployed = root / "user/default/workflows"
                        self.assertEqual(len(list(deployed.glob("*.json"))), 2 + int(compare))
                        self.assertEqual(json.loads((deployed / WORKFLOW).read_text(encoding="utf-8")), build(profile, vae_profile=vae)[1])
                        self.assertEqual(json.loads((deployed / "04_MiniMax_H3_Character_R2V_2x.json").read_text(encoding="utf-8")), prepare(profile, vae_profile=vae)[2])
                        merged = json.loads((root / "user/default/minimax_h3_r2v_models.json").read_text())
                        self.assertEqual(sum(f["path"].startswith("diffusion_models/") for f in merged["files"]), 1)
                        self.assertEqual(sum(f["path"] == "loras/"+LORA for f in merged["files"]), 1)
                        hf = json.loads((root / "user/default/minimax_h3_r2v_hf_models.json").read_text())
                        self.assertFalse(any(f.get("auth") == "civitai" for f in hf["files"]))
            # Official Ref2VA is now the default for 06 only. Keep 04 and
            # text-only 07 on DaSiWa and download shared assets only once.
            (package / "t2va_nodes.py").write_text("# stub")
            env.pop("H3_SWAP_MODEL")
            env.update(H3_R2V_MODEL="dasiwa-v2", H3_T2VA="1")
            result = helper.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((deployed / WORKFLOW).read_text(encoding="utf-8")), build("official", vae_profile="x2-detail")[1])
            self.assertEqual(json.loads((deployed / "04_MiniMax_H3_Character_R2V_2x.json").read_text(encoding="utf-8")), prepare("dasiwa-v2", vae_profile="x2-detail")[2])
            from prepare_t2va import build as build_t2va, WORKFLOW as T2VA_WORKFLOW
            self.assertEqual(json.loads((deployed / T2VA_WORKFLOW).read_text(encoding="utf-8")), build_t2va("dasiwa-v2")[1])
            merged = json.loads((root / "user/default/minimax_h3_r2v_models.json").read_text())
            self.assertEqual(sum(f["path"].startswith("diffusion_models/") for f in merged["files"]), 2)
            self.assertEqual(len({f["path"] for f in merged["files"]}), len(merged["files"]))
            self.assertFalse((root / "models/swap_detection").exists())
            env["H3_SWAP_MODEL"] = "typo"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            env.update(H3_SWAP_MODEL="shared", H3_T2VA="0")
            (package / "swap_nodes.py").unlink()
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            (package / "swap_nodes.py").write_text("# stub")
            personal = deployed / "Personal swap.json"
            personal.write_text("{}")
            env["H3_CHARACTER_SWAP"] = "0"
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((deployed / WORKFLOW).exists())
            self.assertTrue(personal.exists())
            env["H3_CHARACTER_SWAP"] = "typo"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            env.update(H3_CHARACTER_SWAP="1", H3_R2V_MODEL="dasiwa-turbo-v2")
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            env.update(H3_PROFILE="legacy", H3_CHARACTER_R2V="0")
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((deployed / WORKFLOW).exists())


if __name__ == "__main__":
    unittest.main()
