import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import test_character_entrypoint as entrypoint_tests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_t2va import build, WORKFLOW
from prepare_r2v_profile import prepare
from verify_t2va import verify

# Load only the lightweight modules; no ComfyUI/torch required for these tests.
package = types.ModuleType("t2va_unit_package")
package.__path__ = [str(ROOT / "custom_nodes/minimax_h3_ordered_storyboard")]
sys.modules[package.__name__] = package
t2va = importlib.import_module("t2va_unit_package.t2va_nodes")


class TextToVideoTests(unittest.TestCase):
    def test_all_aspects_fit_the_same_pixel_budget_as_06(self):
        swap = importlib.import_module("t2va_unit_package.swap_nodes")
        node = t2va.MiniMaxH3T2VAResolution()
        self.assertEqual(node.dimensions("9:16", .4), (448, 832))
        for ratio in t2va.ASPECT_RATIOS:
            for mp in (.1, .4, .8):
                w, h = node.dimensions(ratio, mp)
                self.assertLessEqual(w*h, mp*1_000_000)
                self.assertEqual(swap.canvas(w, h, mp), (w, h))
        for mp in (0, .9, float("nan"), float("inf")):
            with self.assertRaises(ValueError): node.dimensions("9:16", mp)

    def test_short_text_only_prompt_and_duration(self):
        node = t2va.MiniMaxH3T2VAPrompt()
        self.assertEqual(list(node.INPUT_TYPES()["required"]), ["duration_seconds", "full_prompt"])
        for seconds, frames in ((5., 124), (7., 175), (15., 362)):
            self.assertEqual(node.build(seconds, "```text\nAn adult waves. (gentle:1.2) {literal}\n```"),
                             ("An adult waves. (gentle:1.2) {literal}", frames))
        for text in ("", "  ", "<Picture 1>", "<video 1>", "<Audio 2>", "bad\x00", "x"*16001):
            with self.assertRaises(ValueError): node.build(5, text)
        for seconds in (0, 16, float("inf"), float("nan")):
            with self.assertRaises(ValueError): node.build(seconds, "An adult waves.")

    def test_native_call_cannot_receive_reference_or_vae(self):
        native = Mock()
        native.execute.return_value = ("positive", "latent")
        module = types.ModuleType("comfy_extras.nodes_minimax_h3")
        module.MiniMaxH3ImageToVideo = native
        with patch.dict(sys.modules, {"comfy_extras.nodes_minimax_h3": module}):
            clip = Mock()
            result = t2va.MiniMaxH3TextToVideo().encode(clip, "A woman waves.", 480, 864, 124)
            self.assertEqual(result[:2], ("positive", "latent"))
            self.assertIn("5.167s", result[2])
            call = native.execute.call_args.kwargs
            self.assertIsNone(call["vae"])
            self.assertIsNone(call["first_frame"])
            self.assertIsNone(call["last_frame"])
            self.assertIs(call["clip"].clip, clip)
            for width, height, frames in ((481, 864, 124), (0, 864, 124), (1024, 1024, 124),
                                          (480, 864, 123), (480, 864, 379), (480., 864, 124)):
                with self.assertRaises(ValueError):
                    t2va.MiniMaxH3TextToVideo().encode(clip, "Test", width, height, frames)
            with self.assertRaises(ValueError):
                t2va.MiniMaxH3TextToVideo().encode(clip, "<Picture 1>", 480, 864, 124)
            self.assertEqual(native.execute.call_count, 1)

    def test_graph_models_and_original_workflows_unchanged(self):
        original = prepare("dasiwa-v2", vae_profile="x2-detail")
        manifest, workflow = build("dasiwa-v2")
        verify(workflow, manifest)
        self.assertEqual(prepare("dasiwa-v2", vae_profile="x2-detail"), original)
        self.assertEqual(len(manifest["files"]), 4)
        self.assertFalse(workflow["extra"]["t2va"]["quality_tested_on_gpu"])
        self.assertEqual(build("dasiwa-v2"), (manifest, workflow))
        for profile in ("official", "dasiwa-turbo-v2", "typo"):
            with self.assertRaises(ValueError): build(profile)

    def test_corruption_rejected(self):
        manifest, original = build("dasiwa-v2")
        mutations = (
            lambda w: w["links"][0].__setitem__(3, 9999),
            lambda w: w["links"].append(copy.deepcopy(w["links"][0])),
            lambda w: w["nodes"][0]["outputs"][0]["links"].append(9999),
            lambda w: w["nodes"][0].__setitem__("pos", [-100, 0]),
            lambda w: w["nodes"][0].__setitem__("mode", 4),
            lambda w: next(n for n in w["nodes"] if n["type"] == "UNETLoader")["properties"]["models"].clear(),
            lambda w: next(n for n in w["nodes"] if n["type"] == "BasicScheduler")["widgets_values"].__setitem__(1, 8),
            lambda w: next(n for n in w["nodes"] if n["type"] == "CreateVideo")["widgets_values"].__setitem__(0, 30),
        )
        for mutation in mutations:
            w = copy.deepcopy(original)
            mutation(w)
            with self.assertRaises(RuntimeError): verify(w, manifest)
        bad = copy.deepcopy(manifest)
        bad["files"].pop()
        with self.assertRaises(RuntimeError): verify(original, bad)


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class TextToVideoEntrypointTests(unittest.TestCase):
    def test_opt_in_combinations_download_dedup_and_restore(self):
        helper = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = helper.make_install(temp)
            env.update(H3_PROFILE="r2v", H3_T2VA="1", H3_R2V_MODEL="dasiwa-v2", H3_R2V_VAE="int8")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            package = root / "custom_nodes/minimax_h3_ordered_storyboard"
            for name in ("t2va_nodes.py", "swap_nodes.py", "swap_guard_nodes.py", "compare_nodes.py", "web/upscale_compare.js"):
                (package / name).write_text("# stub")
            decoder = root / "custom_nodes/minimax_h3_x2_vae"
            decoder.mkdir()
            for name in ("__init__.py", "vae_decode.py", "utils.py"):
                (decoder / name).write_text("# stub")
            (decoder / "REVISION").write_text("2e568dfe3e4f5e81da178bc845e05dcfbb64d55b\n")
            for vae in ("int8", "x2-detail"):
                for compare in ("0", "1"):
                    for swap in ("0", "1"):
                        env.update(H3_R2V_VAE=vae, H3_R2V_COMPARE=compare, H3_CHARACTER_SWAP=swap)
                        result = helper.run_entrypoint(env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        directory = root / "user/default"
                        deployed = directory / "workflows"
                        self.assertEqual(len(list(deployed.glob("*.json"))), 2 + int(compare) + int(swap))
                        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
                        self.assertEqual(read(deployed / WORKFLOW), build("dasiwa-v2")[1])
                        self.assertEqual(read(deployed / "04_MiniMax_H3_Character_R2V_2x.json"), prepare("dasiwa-v2", vae_profile=vae)[2])
                        if compare == "1":
                            from prepare_upscale_compare import build as comparison
                            self.assertEqual(read(deployed / "05_MiniMax_H3_Upscale_Compare.json"), comparison("dasiwa-v2")[1])
                        if swap == "1":
                            from prepare_character_swap import build as replacement
                            self.assertEqual(read(deployed / "06_MiniMax_H3_Character_Swap.json"), replacement("dasiwa-v2", vae_profile=vae)[1])
                        merged = read(directory / "minimax_h3_r2v_models.json")
                        paths = [f["path"] for f in merged["files"]]
                        self.assertEqual(len(paths), len(set(paths)))
                        self.assertEqual(sum(p.startswith("diffusion_models/") for p in paths), 1)
                        self.assertEqual(paths.count("vae/minimax_h3_video_vae_int8_convrot.safetensors"), 1)
                        hf = read(directory / "minimax_h3_r2v_hf_models.json")
                        self.assertFalse(any(f.get("auth") == "civitai" for f in hf["files"]))
                        self.assertEqual(hf["total_bytes"], sum(f["size"] for f in hf["files"]))
            personal = deployed / "My text video.json"
            personal.write_text("{}")
            env["H3_T2VA"] = "0"
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((deployed / WORKFLOW).exists())
            self.assertTrue(personal.exists())
            env["H3_T2VA"] = "oops"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            for profile in ("official", "dasiwa-turbo-v2"):
                env.update(H3_T2VA="1", H3_R2V_MODEL=profile, H3_CHARACTER_SWAP="0")
                self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            env.update(H3_PROFILE="legacy", H3_CHARACTER_R2V="0")
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((deployed / WORKFLOW).exists())


if __name__ == "__main__":
    unittest.main()
