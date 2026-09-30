"""Prompt/graph security contracts; real RefMod roundtrips when CPU torch is installed."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refmod_test", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/refmod_nodes.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_r2v_profile
import verify_workflow


class PromptAndBudgetTests(unittest.TestCase):
    def test_one_full_field_preserves_content_not_four_fields(self):
        required = ref.MiniMaxH3FullPrompt.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["duration_seconds", "full_prompt"])
        raw = "subject_definitions:\n<Subject 1> from <Picture 1>.\nnon_diegetic_music:\nBGM: drums!"
        for text in (raw, "\ufeff" + raw, "```text\n" + raw + "\n```", raw.replace("\n", "\r\n")):
            self.assertEqual(ref.MiniMaxH3FullPrompt().build(5, text), (raw, 124))
        self.assertEqual(ref.clean_full_prompt("A wave. (gentle:1.2) {no random variants}"),
                         "A wave. (gentle:1.2) {no random variants}")
        for bad in (None, " ", "```\n\n```", "\x00bad", "a" * 16001):
            with self.assertRaises(ValueError): ref.clean_full_prompt(bad)
        for duration in (0, 30, float("inf"), float("nan")):
            with self.assertRaises(ValueError): ref.MiniMaxH3FullPrompt().build(duration, raw)

    def test_full_reference_numbers_fail_before_model_execution(self):
        ref.validate_references("<Picture 1> <Picture 3> <Subject 1>", 3)
        for text in ("<Picture 0>", "<Picture 2>", "<Video 1>", "<Audio 1>"):
            with self.assertRaises(ValueError): ref.validate_references(text, 1)

    def test_text_token_limit_checks_before_encoder_without_global_patch(self):
        clip = mock.Mock()
        original = "A wave. (gentle:1.2) {literal text}"
        tokens = {"qwen3vl_32b": [[(5, 1.0)] * 4096 + [({"type": "image"}, 1.0)]]}
        clip.tokenize.return_value = tokens
        guarded = ref.PromptBudgetClip(clip)
        self.assertIs(guarded.tokenize(original, ref_data=[]), tokens)
        clip.tokenize.assert_called_once_with(original, ref_data=[])
        self.assertIs(guarded.encode_from_tokens_scheduled(tokens), clip.encode_from_tokens_scheduled.return_value)
        clip.encode_from_tokens_scheduled.assert_called_once_with(tokens)
        clip.encode_from_tokens_scheduled.reset_mock()
        tokens["qwen3vl_32b"][0].append((6, 1.0))
        with self.assertRaisesRegex(ValueError, "4096"):
            guarded.tokenize(original)
        clip.encode_from_tokens_scheduled.assert_not_called()

    def test_one_to_eight_sizes_bounded_without_crop_or_upscale(self):
        for sizes in ([(1728, 2304)], [(1728, 2304), (2048, 1024), (512, 512)], [(4096, 8192)] * 8):
            for budget in (256, 2048, 4096):
                result = ref.reference_sizes(sizes, 1024, budget)
                self.assertEqual(len(result), len(sizes))
                self.assertLessEqual(sum(w * h // 1024 for w, h in result), budget)
                for (w, h), (ow, oh) in zip(result, sizes):
                    self.assertLessEqual(w, ow); self.assertLessEqual(h, oh)
                    self.assertLessEqual(max(w, h), 1024)
                    self.assertEqual(w % 32, 0); self.assertEqual(h % 32, 0)
        for args in (([], 1024, 2048), ([(512, 512)] * 9, 1024, 2048), ([(1, 512)], 1024, 2048),
                     ([(512, 512)], 2048, 2048), ([(512, 512)], 1024, 0)):
            with self.assertRaises(ValueError): ref.reference_sizes(*args)

    def test_input_path_validation_and_ephemeral_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"; root.mkdir()
            (root / "one.png").touch(); (Path(temp) / "secret.png").touch()
            self.assertEqual(ref.image_paths('["one.png"]', root), [root / "one.png"])
            for bad in ('[]', '{}', 'null', 'broken', '["../secret.png"]', '["C:/private.png"]',
                        '["one.png", "one.png"]', '["missing.png"]', '[3]'):
                with self.assertRaises(ValueError): ref.image_paths(bad, root)

    def test_missing_refmod_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"MINIMAX_H3_REFMOD_ROOT": temp}):
            with self.assertRaises(RuntimeError): ref.refmod_modules()
            (Path(temp) / "REVISION").write_text("main")
            with self.assertRaises(RuntimeError): ref.refmod_modules()

    def test_refmod_graph_contract_and_corruption_detection(self):
        for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
            _, _, workflow = prepare_r2v_profile.prepare(profile)
            verify_workflow.verify_character_refmod(workflow)
            for kind in ("MiniMaxH3CreateCharacterRefMod", "MiniMaxH3FullPrompt", "MiniMaxH3RefModImages"):
                bad = copy.deepcopy(workflow)
                bad["nodes"] = [n for n in bad["nodes"] if n["type"] != kind]
                with self.assertRaises(RuntimeError): verify_workflow.verify_character_refmod(bad)
            bad = copy.deepcopy(workflow)
            encode = next(n for n in bad["nodes"] if n["type"] == "MiniMaxH3CharacterRefModR2V")
            bad["links"] = [l for l in bad["links"] if l[3] != encode["id"]]
            with self.assertRaises(RuntimeError): verify_workflow.verify_character_refmod(bad)


HAS_RUNTIME = bool(importlib.util.find_spec("torch")) and bool(os.environ.get("MINIMAX_H3_REFMOD_ROOT"))


@unittest.skipUnless(HAS_RUNTIME, "CPU RefMod suite runs in its venv/CI, and real native API smoke runs in Docker")
class RefModCPUTests(unittest.TestCase):
    def test_real_upstream_roundtrip_and_second_run_cpu_payload(self):
        import torch
        from PIL import Image
        core, bundle = ref.refmod_modules()
        class H3VAE: pass
        class VAE:
            first_stage_model = H3VAE()
            def encode(self, image):
                return torch.full((1, 24, 1, image.shape[1] // 16, image.shape[2] // 16),
                                  float(image.mean()), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (384, 512), "red").save(root / "a.png")
            Image.new("RGB", (512, 256), "blue").save(root / "b.png")
            folders = SimpleNamespace(get_input_directory=lambda: temp, get_output_directory=lambda: temp)
            native_calls = []
            class Native:
                @classmethod
                def execute(cls, *args, **kwargs):
                    native_calls.append((args, kwargs))
                    return SimpleNamespace(result=([[torch.zeros(1), {"minimax_token_tags": torch.ones(1)}]], {"samples": "fixture"}))
            with mock.patch.dict(sys.modules, {"folder_paths": folders,
                    "comfy.ldm.minimax.vae": SimpleNamespace(MiniMaxH3VideoVAE=H3VAE),
                    "comfy_extras.nodes_minimax_h3": SimpleNamespace(MiniMaxH3ReferenceToVideo=Native)}):
                for names in (["a.png"], ["a.png", "b.png"], ["b.png"]):
                    refs = ref.MiniMaxH3RefModImages().load(json.dumps(names), 1024, 2048)[0]
                    character, mods = ref.MiniMaxH3CreateCharacterRefMod().create(refs, VAE())
                    self.assertEqual(len(mods), len(names))
                    for mod, weight in mods:
                        self.assertEqual(mod.kind, "image"); self.assertEqual(weight, 1.0)
                        self.assertEqual(mod.latent.device.type, "cpu")
                    output = ref.MiniMaxH3CharacterRefModR2V().encode(object(), character,
                        "<Subject 1> from <Picture 1> waves. non_diegetic_music: N/A", 480, 864, 124)
                    self.assertEqual(len(output[0][0][1]["minimax_refs"]), len(names))
                    self.assertIsNone(native_calls[-1][1]["vae"])
                    self.assertEqual(len(native_calls[-1][1]["ref_images"]), len(names))
                    self.assertNotIn("minimax_refs", {"minimax_token_tags": torch.ones(1)})
                    saved = ref.MiniMaxH3SaveCharacterRefMod().save(mods)["result"][0]
                    loaded = bundle.load_bundle(str(Path(saved).with_suffix("")))
                    self.assertEqual(len(loaded), len(mods))
                    for (left, _), (right, _) in zip(mods, loaded):
                        self.assertTrue(torch.equal(left.latent, right.latent))
                        self.assertEqual(left.token_count, right.token_count)
                count = len(native_calls)
                with self.assertRaises(ValueError):
                    ref.MiniMaxH3CharacterRefModR2V().encode(object(), character, "<Picture 2>", 480, 864, 124)
                self.assertEqual(len(native_calls), count)
                with self.assertRaises(ValueError): ref.MiniMaxH3SaveCharacterRefMod().save(mods, "../bad")

    def test_content_change_invalidates_loader_cache(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.png"
            with mock.patch.dict(sys.modules, {"folder_paths": SimpleNamespace(get_input_directory=lambda: temp)}):
                Image.new("RGB", (32, 32), "red").save(path)
                a = ref.MiniMaxH3RefModImages.IS_CHANGED('["a.png"]')
                Image.new("RGB", (32, 32), "blue").save(path)
                self.assertNotEqual(a, ref.MiniMaxH3RefModImages.IS_CHANGED('["a.png"]'))


if __name__ == "__main__": unittest.main()
