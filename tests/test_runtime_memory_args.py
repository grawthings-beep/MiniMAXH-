"""Exercise startup migration before the license gate, without GPUs or downloads."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if not BASH and os.name == "nt":
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    if git_bash.exists():
        BASH = str(git_bash)


@unittest.skipUnless(BASH, "bash is required for entrypoint execution")
class RuntimeMemoryArgsTests(unittest.TestCase):
    def test_old_defaults_upgrade_but_custom_profiles_are_preserved(self):
        stable = "--lowvram --vram-headroom 2 --cache-none --disable-smart-memory"
        profiles = {
            "--lowvram --vram-headroom 2": stable,
            "--disable-dynamic-vram --reserve-vram 4": stable,
            stable: stable,
            "--lowvram --vram-headroom 4 --cache-classic":
                "--lowvram --vram-headroom 4 --cache-classic",
        }
        with tempfile.TemporaryDirectory() as directory:
            # Windows Git checkouts may use CRLF; execute the Linux image's LF content.
            script = Path(directory) / "entrypoint.sh"
            script.write_text((ROOT / "scripts/entrypoint.sh").read_text(encoding="utf-8"),
                              encoding="utf-8", newline="\n")
            for before, expected in profiles.items():
                with self.subTest(profile=before):
                    env = dict(os.environ, COMFYUI_ARGS=before, ACCEPT_MINIMAX_H3_LICENSE="0")
                    result = subprocess.run(
                        [BASH, "-c", 'trap \'printf "PROFILE_RESULT=%s\\n" "$COMFYUI_ARGS"\' EXIT; source "$1"',
                         "memory-profile-test", script.as_posix()],
                        env=env, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 64, result.stderr + result.stdout)
                    self.assertIn(f"PROFILE_RESULT={expected}\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
