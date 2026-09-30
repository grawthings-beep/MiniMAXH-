"""Visible R2VA LoRA controls, bypass/cache behavior and opt-in startup selection."""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from test_turbo_profile import turbo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_r2v_loras as downloads


class R2VLoRATests(unittest.TestCase):
    def test_schema_shows_only_installed_files_and_explicit_off(self):
        paths = types.SimpleNamespace(get_filename_list=lambda _kind: ["custom.safetensors", turbo.TURBO_8STEP_MODEL])
        with mock.patch.dict(sys.modules, {"folder_paths": paths}):
            required = turbo.MiniMaxH3R2VLoRA.INPUT_TYPES()["required"]
            self.assertEqual(required["lora_name"][0], [turbo.R2V_LORA_NONE, "custom.safetensors"])
            self.assertEqual(required["strength"][1]["default"], 0.4)
            self.assertFalse(required["enabled"][1]["default"])
            self.assertTrue(turbo.MiniMaxH3R2VLoRA.VALIDATE_INPUTS("custom.safetensors", 0.4, True))
            self.assertIsInstance(turbo.MiniMaxH3R2VLoRA.VALIDATE_INPUTS("missing.safetensors", 0.4, True), str)
        self.assertIs(turbo.NODE_CLASS_MAPPINGS["MiniMaxH3R2VLoRA"], turbo.MiniMaxH3R2VLoRA)

    def test_off_and_zero_need_no_file_or_comfy_import_and_clear_cache(self):
        node, model = turbo.MiniMaxH3R2VLoRA(), object()
        with mock.patch.dict(sys.modules, {"folder_paths": None, "comfy": None}):
            for enabled, strength in [(False, 0.4), (True, 0.0)]:
                node._loaded_lora = ("old", object(), {})
                self.assertTrue(node.VALIDATE_INPUTS("missing.safetensors", strength, enabled))
                self.assertIs(node.apply_lora(model, "missing.safetensors", strength, enabled)[0], model)
                self.assertIsNone(node._loaded_lora)

    def test_on_unselected_or_invalid_strength_fails_without_model_load(self):
        node = turbo.MiniMaxH3R2VLoRA()
        for name, strength in [(turbo.R2V_LORA_NONE, 0.4), (turbo.TURBO_8STEP_MODEL, 1.0),
                               ("custom", float("nan")), ("custom", float("inf")), ("custom", 5.0)]:
            with self.assertRaises(ValueError):
                node.apply_lora(object(), name, strength, True)

    def test_on_reuses_only_current_file_metadata_and_off_restores_original(self):
        loads, patches = [], []
        paths = types.SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(Path("/models") / kind / name))
        def load(path, **kwargs):
            self.assertTrue(kwargs["safe_load"])
            self.assertTrue(kwargs["return_metadata"])
            loads.append(path)
            return {"file": path}, {"fixture": "metadata"}
        def patch(model, clip, lora, strength, clip_strength, **kwargs):
            self.assertIsNone(clip)
            self.assertEqual(clip_strength, 0.0)
            self.assertEqual(kwargs["lora_metadata"], {"fixture": "metadata"})
            patches.append((model, lora, strength))
            return object(), None
        comfy = types.SimpleNamespace(utils=types.SimpleNamespace(load_torch_file=load),
                                      sd=types.SimpleNamespace(load_lora_for_models=patch))
        node, model = turbo.MiniMaxH3R2VLoRA(), object()
        with mock.patch.dict(sys.modules, {"folder_paths": paths, "comfy": comfy}):
            for name, strength in [("a.safetensors", 0.4), ("a.safetensors", 0.2), ("b.safetensors", 0.3)]:
                self.assertIsNot(node.apply_lora(model, name, strength, True)[0], model)
            self.assertEqual(len(loads), 2)
            self.assertTrue(all(p[0] is model for p in patches))  # Never patch the previous patched output.
            self.assertIs(node.apply_lora(model, "b.safetensors", 0.3, False)[0], model)
            self.assertIsNone(node._loaded_lora)
            node.apply_lora(model, "a.safetensors", 0.4, True)
            self.assertEqual(len(loads), 3)


class R2VDownloadSelectionTests(unittest.TestCase):
    def test_legacy_flags_do_not_enable_downloads(self):
        ids, _ = downloads.download_env({"H3_LORA_SELECTION": "all", "H3_LORA_REQUIRED": "1"})
        self.assertEqual(ids, ())
        for value in ("typo", "none,all", "hmnsfw_aio_v2,", "all,hmmotion_v1"):
            with self.assertRaises(ValueError):
                downloads.selection({"H3_R2V_LORA_SELECTION": value})

    def test_selection_requires_credentials_and_preserves_source_overrides(self):
        env = {"H3_R2V_LORA_SELECTION": " HMMOTION_V1, hmnsfw_aio_v2 ", "H3_LORA_REQUIRED": "0",
               "HF_TOKEN": "hf-secret", "CIVITAI_API_TOKEN": "api-secret", "H3_LORA_REVISION": "pinned"}
        ids, child = downloads.download_env(env)
        self.assertEqual(set(ids), set(downloads.SCRIPTS))
        self.assertEqual(child["H3_LORA_REQUIRED"], "1")
        self.assertEqual(child["CIVITAI_TOKEN"], "api-secret")
        self.assertEqual(child["H3_LORA_REVISION"], "pinned")
        self.assertEqual(env["H3_LORA_REQUIRED"], "0")
        for token in ("HF_TOKEN", "CIVITAI_API_TOKEN"):
            with self.assertRaises(ValueError):
                downloads.download_env({k: v for k, v in env.items() if k != token})

    def test_check_mode_does_not_start_transfers(self):
        with mock.patch.dict(os.environ, {"H3_R2V_LORA_SELECTION": "none"}, clear=True), \
             mock.patch.object(sys, "argv", ["download_r2v_loras.py", "--check"]), \
             mock.patch.object(downloads.subprocess, "Popen") as popen:
            self.assertEqual(downloads.main(), 0)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
