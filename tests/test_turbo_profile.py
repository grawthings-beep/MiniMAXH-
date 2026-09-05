from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "custom_nodes"
    / "minimax_h3_ordered_storyboard"
    / "turbo_nodes.py"
)
SPEC = importlib.util.spec_from_file_location("minimax_h3_turbo_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
turbo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = turbo
SPEC.loader.exec_module(turbo)


class TurboProfileTests(unittest.TestCase):
    def test_profile_schema_and_validation(self):
        choices = turbo.MiniMaxH3TurboProfile.INPUT_TYPES()["required"]["profile"][0]
        self.assertEqual(
            choices,
            (turbo.PROFILE_8STEP, turbo.PROFILE_4STEP),
        )
        self.assertTrue(
            turbo.MiniMaxH3TurboProfile.VALIDATE_INPUTS(turbo.PROFILE_8STEP)
        )
        self.assertIsInstance(
            turbo.MiniMaxH3TurboProfile.VALIDATE_INPUTS("invalid"), str
        )

    def test_selection_loads_matching_lora_and_emits_matching_steps(self):
        loads = []
        patches = []

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = (
            lambda category, name: str(Path("/models") / category / name)
        )
        comfy = types.ModuleType("comfy")
        comfy.utils = types.SimpleNamespace(
            load_torch_file=lambda path, **kwargs: (
                loads.append((path, kwargs)) or {"weight": path},
                {"source": path},
            )
        )

        def apply_lora(model, clip, lora, strength_model, strength_clip, **kwargs):
            patches.append(
                (model, clip, lora, strength_model, strength_clip, kwargs)
            )
            return f"{model}:{Path(lora['weight']).name}", clip

        comfy.sd = types.SimpleNamespace(load_lora_for_models=apply_lora)

        selector = turbo.MiniMaxH3TurboProfile()
        with mock.patch.dict(
            sys.modules,
            {"folder_paths": folder_paths, "comfy": comfy},
        ):
            model_8, steps_8 = selector.apply_profile("base", turbo.PROFILE_8STEP)
            cached_8, cached_steps = selector.apply_profile("base", turbo.PROFILE_8STEP)
            model_4, steps_4 = selector.apply_profile("base", turbo.PROFILE_4STEP)

        self.assertTrue(model_8.endswith(turbo.TURBO_8STEP_MODEL))
        self.assertEqual((steps_8, cached_steps), (8, 8))
        self.assertEqual(model_8, cached_8)
        self.assertTrue(model_4.endswith(turbo.TURBO_4STEP_MODEL))
        self.assertEqual(steps_4, 4)
        self.assertEqual(len(loads), 2, "same active profile should reuse its state dict")
        self.assertEqual(len(patches), 3)
        self.assertTrue(all(call[3:5] == (1.0, 0.0) for call in patches))
        self.assertTrue(
            str(selector._loaded_lora[0]).endswith(turbo.TURBO_4STEP_MODEL)
        )

    def test_unknown_profile_fails_before_loading_comfy(self):
        selector = turbo.MiniMaxH3TurboProfile()
        with self.assertRaisesRegex(ValueError, "Unknown MiniMax H3 Turbo profile"):
            selector.apply_profile("base", "invalid")


if __name__ == "__main__":
    unittest.main()
