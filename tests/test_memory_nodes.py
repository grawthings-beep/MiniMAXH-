from __future__ import annotations

import importlib.util
import gc
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
    / "memory_nodes.py"
)
SPEC = importlib.util.spec_from_file_location("minimax_h3_memory_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = memory
SPEC.loader.exec_module(memory)


class MemoryNodeTests(unittest.TestCase):
    def test_unreachable_cycles_are_collected_before_allocator_cleanup_each_run(self):
        collected = []
        generation = []

        class TensorOwner:
            def __init__(self, run):
                self.run = run
                self.cycle = self

            def __del__(self):
                collected.append(self.run)

        def unload():
            generation.append(len(generation) + 1)
            TensorOwner(generation[-1])

        def empty_cache(force):
            self.assertTrue(force)
            self.assertIn(generation[-1], collected,
                          "unreachable tensor owners must be released before empty_cache")

        comfy = types.ModuleType("comfy")
        comfy.model_management = types.SimpleNamespace(
            unload_all_models=unload, soft_empty_cache=empty_cache,
        )
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with mock.patch.dict(sys.modules, {"comfy": comfy}):
                for _ in range(2):
                    samples = {"samples": object()}
                    self.assertIs(memory.MiniMaxH3ReleaseVRAMLatent().release(samples)[0], samples)
            self.assertEqual(collected, [1, 2])
        finally:
            gc.collect()
            if was_enabled:
                gc.enable()

    def test_release_unloads_models_and_forces_allocator_cleanup(self):
        calls = []
        comfy = types.ModuleType("comfy")
        comfy.model_management = types.SimpleNamespace(
            unload_all_models=lambda: calls.append(("unload",)),
            soft_empty_cache=lambda force=False: calls.append(("empty", force)),
        )
        samples = {"samples": object()}
        with mock.patch.dict(sys.modules, {"comfy": comfy}):
            output = memory.MiniMaxH3ReleaseVRAMLatent().release(samples)
        self.assertIs(output[0], samples)
        self.assertEqual(calls, [("unload",), ("empty", True)])

    def test_tiled_decode_selects_video_from_nested_h3_latent(self):
        video = object()
        audio = object()

        class NestedLatent:
            is_nested = True

            def unbind(self):
                return video, audio

        decoded = mock.MagicMock()
        decoded.shape = (1, 3, 2, 64, 96)
        flattened = object()
        decoded.reshape.return_value = flattened
        vae = mock.MagicMock()
        vae.decode_tiled.return_value = decoded

        result = memory.MiniMaxH3VAEDecodeTiled().decode(
            vae,
            {"samples": NestedLatent()},
        )

        vae.decode_tiled.assert_called_once_with(video)
        decoded.reshape.assert_called_once_with(-1, 2, 64, 96)
        self.assertIs(result[0], flattened)


if __name__ == "__main__":
    unittest.main()
