from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
SPEC = importlib.util.spec_from_file_location("preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class ProfileTests(unittest.TestCase):
    def test_profiles(self) -> None:
        self.assertEqual(preflight.profile_for_vram(7.9)["name"], "unsupported")
        self.assertEqual(preflight.profile_for_vram(12)["name"], "preview")
        self.assertEqual(preflight.profile_for_vram(16)["name"], "balanced")
        self.assertEqual(preflight.profile_for_vram(24)["name"], "high")
        self.assertEqual(preflight.profile_for_vram(32)["name"], "native-768p")

    def test_gpu_query_falls_back_to_torch_when_nvml_memory_is_restricted(self) -> None:
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _device: types.SimpleNamespace(
                name="NVIDIA GeForce RTX 4090",
                total_memory=24 * preflight.GIB,
            ),
        )
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)
        restricted = (
            "NVIDIA GeForce RTX 4090, [Insufficient Permissions], 580.95.05\n"
        )

        with (
            mock.patch.object(preflight.subprocess, "check_output", return_value=restricted),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            name, vram_gib, driver = preflight.query_gpu()

        self.assertEqual(name, "NVIDIA GeForce RTX 4090")
        self.assertEqual(vram_gib, 24)
        self.assertEqual(driver, "580.95.05")

    def test_gpu_query_reports_both_probe_failures(self) -> None:
        fake_cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)

        with (
            mock.patch.object(
                preflight.subprocess,
                "check_output",
                return_value="GPU, [Insufficient Permissions], 580.95.05\n",
            ),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "nvidia-smi or PyTorch"
            ):
                preflight.query_gpu()


if __name__ == "__main__":
    unittest.main()
