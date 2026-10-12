"""08 audio-only graph, opt-in startup, bounded input and no-reference prompt."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import test_character_entrypoint as entrypoint_tests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_r2v_profile import prepare
from prepare_video_audio import WORKFLOW, build
from verify_video_audio import verify

package = types.ModuleType("video_audio_unit_package")
package.__path__ = [str(ROOT / "custom_nodes/minimax_h3_ordered_storyboard")]
sys.modules[package.__name__] = package
audio = importlib.import_module("video_audio_unit_package.video_audio_nodes")


class VideoAudioTests(unittest.TestCase):
    def test_aligned_input_and_prompt(self):
        self.assertEqual(audio.aligned_frames(5.), 124)
        self.assertEqual(audio.aligned_frames(5.167), 124)
        self.assertEqual(audio.aligned_frames(5.2), 141)
        self.assertEqual(audio.aligned_frames(15.), 362)
        for duration in (4.4, 15.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): audio.aligned_frames(duration)
        prompt = audio.MiniMaxH3VideoAudioPrompt()
        self.assertEqual(prompt.build("Natural sounds of <Video 1>. No dialogue."),
                         ("Natural sounds of <Video 1>. No dialogue.",))
        self.assertIn("<Video 1>", prompt.build("Soft footsteps.")[0])
        for invalid in ("<Picture 1>", "<Audio 1>", "<Video 2>", "", "bad\x00"):
            with self.assertRaises(ValueError): prompt.build(invalid)

    def test_graph_manifest_and_04_unchanged(self):
        for profile in ("official", "dasiwa-v2"):
            before = prepare(profile, vae_profile="x2-detail")
            manifest, workflow = build(profile)
            verify(workflow, manifest)
            self.assertEqual(build(profile), (manifest, workflow))
            self.assertEqual(prepare(profile, vae_profile="x2-detail"), before)
            self.assertFalse(workflow["extra"]["video_audio"]["quality_tested_on_gpu"])
            self.assertEqual(len(manifest["files"]), 4)
            types_ = [n["type"] for n in workflow["nodes"]]
            self.assertEqual(types_.count("VAEDecodeAudio"), 1)
            self.assertEqual(types_.count("MiniMaxH3VideoAudioSave"), 1)
            self.assertNotIn("VAEDecode", types_)
            self.assertNotIn("CreateVideo", types_)
            self.assertNotIn("WanAutoMosaicVideo", types_)
        with self.assertRaises(ValueError): build("dasiwa-turbo-v2")

    def test_corrupt_graph_rejected(self):
        manifest, workflow = build("dasiwa-v2")
        for mutate in (
            lambda w: w["links"][0].__setitem__(1, 9999),
            lambda w: w["links"][0].__setitem__(5, "AUDIO"),
            lambda w: w["nodes"][0].__setitem__("mode", 4),
            lambda w: next(n for n in w["nodes"] if n["type"] == "MiniMaxH3VideoAudioSave")["inputs"][1].__setitem__("link", None),
            lambda w: next(n for n in w["nodes"] if n["type"] == "VAELoader" and "video_vae" in n["widgets_values"][0])["widgets_values"].__setitem__(0, "wrong.safetensors"),
        ):
            changed = copy.deepcopy(workflow)
            mutate(changed)
            with self.assertRaises((RuntimeError, IndexError)):
                verify(changed, manifest)


@unittest.skipIf(os.name == "nt", "Git Bash shared-memory namespace is blocked by the Windows sandbox")
@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class VideoAudioEntrypointTests(unittest.TestCase):
    def test_opt_in_manifest_and_disable(self):
        helper = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temporary:
            root, env = helper.make_install(temporary)
            env.update(H3_PROFILE="r2v", H3_VIDEO_AUDIO="1", H3_R2V_MODEL="dasiwa-v2",
                       H3_R2V_VAE="x2-detail", H3_T2VA="0")
            self.assertEqual(helper.run_entrypoint(env).returncode, 77)
            (root / "custom_nodes/minimax_h3_ordered_storyboard/video_audio_nodes.py").write_text("# stub")
            decoder = root / "custom_nodes/minimax_h3_x2_vae"
            decoder.mkdir()
            for name in ("__init__.py", "vae_decode.py", "utils.py"):
                (decoder / name).write_text("# stub")
            (decoder / "REVISION").write_text("2e568dfe3e4f5e81da178bc845e05dcfbb64d55b\n")
            result = helper.run_entrypoint(env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            installed = root / "user/default"
            deployed = json.loads((installed / "workflows" / WORKFLOW).read_text(encoding="utf-8"))
            self.assertEqual(deployed, build("dasiwa-v2")[1])
            paths = [f["path"] for f in json.loads((installed / "minimax_h3_r2v_models.json").read_text())["files"]]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertIn("vae/minimax_h3_video_vae_int8_convrot.safetensors", paths)
            self.assertIn("vae/MiniMax-H3-X2-Detail-v1.safetensors", paths)
            env["H3_VIDEO_AUDIO"] = "0"
            self.assertEqual(helper.run_entrypoint(env).returncode, 0)
            self.assertFalse((installed / "workflows" / WORKFLOW).exists())
            env["H3_VIDEO_AUDIO"] = "bad"
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)
            env.update(H3_VIDEO_AUDIO="1", H3_R2V_MODEL="dasiwa-turbo-v2")
            self.assertEqual(helper.run_entrypoint(env).returncode, 79)


if __name__ == "__main__":
    unittest.main()
