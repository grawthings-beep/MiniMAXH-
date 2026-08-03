from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()

