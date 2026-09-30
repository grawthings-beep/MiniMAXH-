"""Selected-only models, sampling/link/layout contracts, secret-safe resume."""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import test_character_entrypoint as entrypoint_tests
import test_character_r2v as character_tests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_r2v_profile as profiles
import download_civitai_models as downloader
import verify_workflow as verifier


class ProfileTests(unittest.TestCase):
    def test_official_unchanged_and_bad_choice_rejected(self):
        manifest, companions, workflow = profiles.prepare("official")
        self.assertEqual(manifest, companions)
        self.assertEqual(workflow, json.loads((ROOT / "workflows/character_reveal_r2v_int8_2x.json").read_text(encoding="utf-8")))
        with self.assertRaises(ValueError):
            profiles.prepare("latest")

    def test_pinned_files_replace_not_add_and_leave_companions_unchanged(self):
        official, _, _ = profiles.prepare("official")
        expected_common = [f for f in official["files"] if not f["path"].startswith("diffusion_models/")]
        for profile, size, sha in [
            ("dasiwa-v2", 20967669160, "4cb8e1eaa9c3e5c664822760890bcbe6078455401cfa43205baf322e856d25f8"),
            ("dasiwa-turbo-v2", 20967669168, "37c17fd91971c17e02e60798a05eec48d881e1bb970d6309f58cf134a2d03a6b"),
        ]:
            manifest, common, workflow = profiles.prepare(profile)
            self.assertEqual(common["files"], expected_common)
            self.assertEqual(len(manifest["files"]), 5)
            self.assertEqual(manifest["revision"], official["revision"])
            self.assertEqual(manifest["total_bytes"], sum(f["size"] for f in manifest["files"]))
            self.assertEqual(common["total_bytes"], sum(f["size"] for f in common["files"]))
            asset = manifest["files"][0]
            downloader.validate_asset(asset)
            self.assertEqual((asset["size"], asset["sha256"]), (size, sha))
            loader = next(n for n in workflow["nodes"] if n["type"] == "UNETLoader")
            self.assertEqual(loader["widgets_values"], [Path(asset["path"]).name, "default"])
            self.assertEqual(loader["properties"]["models"][0]["url"], asset["source_url"])
            self.assertEqual(len([f for f in manifest["files"] if f["path"].startswith("diffusion_models/")]), 1)

    def test_sampler_shift_both_consumers_and_unchanged_user_inputs_tail(self):
        _, _, official = profiles.prepare("official")
        original = {n["type"]: n for n in official["nodes"]}
        for profile, sampler, steps, shifts in [
            ("dasiwa-v2", "res_multistep", 25, [11.0, 4.0]),
            ("dasiwa-turbo-v2", "euler", 8, [8.0, 4.0]),
        ]:
            _, _, workflow = profiles.prepare(profile)
            nodes = {n["type"]: n for n in workflow["nodes"]}
            self.assertEqual(nodes["BasicScheduler"]["widgets_values"], ["simple", steps, 1.0])
            self.assertEqual(nodes["KSamplerSelect"]["widgets_values"], [sampler])
            shift = nodes["MiniMaxH3SigmaShift"]
            self.assertEqual(shift["widgets_values"], shifts)
            targets = {l[3] for l in workflow["links"] if l[1] == shift["id"]}
            self.assertEqual(targets, {nodes["BasicScheduler"]["id"], nodes["BasicGuider"]["id"]})
            lora = nodes["MiniMaxH3R2VLoRA"]
            self.assertEqual([l[3] for l in workflow["links"] if l[1] == nodes["UNETLoader"]["id"]], [lora["id"]])
            self.assertEqual([l[3] for l in workflow["links"] if l[1] == lora["id"]], [shift["id"]])
            self.assertEqual(lora["widgets_values"], ["None (select an installed H3 LoRA)", 0.4, False])
            for kind in ("MiniMaxH3RefModImages", "MiniMaxH3FullPrompt", "MiniMaxH3CreateCharacterRefMod", "MiniMaxH3CharacterRefModR2V",
                         "ResolutionSelector", "MiniMaxH3ReleaseVRAMLatent", "MiniMaxH3VAEDecodeTiled",
                         "VAEDecodeAudio", "UpscaleModelLoader", "ImageUpscaleWithModel", "WanAutoMosaicVideo", "CreateVideo"):
                self.assertEqual(nodes[kind]["widgets_values"], original[kind]["widgets_values"])
            self.assertNotIn("definitions", workflow)
            self.assertEqual(len([n for n in workflow["nodes"] if n["type"] == "WanAutoMosaicVideo"]), 1)
            self.assertFalse(any("Lora" in t or "LoRA" in t or "Cache" in t for t in nodes if t != "MiniMaxH3R2VLoRA"))
            # Reuse actual geometric + bidirectional link assertions, not a string check.
            checker = character_tests.CharacterTests()
            checker.workflow = workflow
            checker.test_bidirectional_links_types_and_topological_order()
            checker.test_no_node_or_group_overlap_and_every_node_is_contained()

    def test_all_generated_profiles_pass_existing_dependency_and_tail_verifier(self):
        for profile in ("official", "dasiwa-v2", "dasiwa-turbo-v2"):
            with tempfile.TemporaryDirectory() as temp:
                subprocess.run([sys.executable, str(ROOT / "scripts/prepare_r2v_profile.py"),
                                "--profile", profile, "--output-dir", temp], check=True, capture_output=True)
                result = subprocess.run([
                    sys.executable, str(ROOT / "scripts/verify_workflow.py"),
                    "--workflow", str(Path(temp) / "workflows" / profiles.WORKFLOW),
                    "--manifest", str(Path(temp) / "minimax_h3_r2v_models.json"), "--mode", "r2v",
                    "--expect-upscale", "--expect-memory-safe-decode", "--expect-auto-mosaic",
                    "--auto-mosaic-manifest", str(ROOT / "manifests/auto_mosaic.json"),
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_verifier_rejects_wrong_model_shift_sampler_or_bypassed_path(self):
        manifest, _, workflow = profiles.prepare("dasiwa-v2")
        for kind, values in [("BasicScheduler", ["normal", 25, 1.0]), ("KSamplerSelect", ["euler"]),
                             ("MiniMaxH3SigmaShift", [12.0, 3.0]), ("UNETLoader", ["wrong.safetensors", "default"])]:
            bad = copy.deepcopy(workflow)
            next(n for n in bad["nodes"] if n["type"] == kind)["widgets_values"] = values
            with self.assertRaises(RuntimeError):
                verifier.verify_dasiwa_profile(bad, manifest)
        bad = copy.deepcopy(workflow)
        unet = next(n for n in bad["nodes"] if n["type"] == "UNETLoader")
        guide = next(n for n in bad["nodes"] if n["type"] == "BasicGuider")
        next(l for l in bad["links"] if l[3] == guide["id"] and l[5] == "MODEL")[1] = unet["id"]
        with self.assertRaises(RuntimeError):
            verifier.verify_dasiwa_profile(bad, manifest)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        header = json.dumps({"tensor": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
        self.payload = len(header).to_bytes(8, "little") + header + b"test"
        self.asset = {"path": "diffusion_models/test.safetensors", "auth": "civitai",
                      "source_url": "https://civitai.com/api/download/models/3314675?fileId=3203130",
                      "size": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}

    def test_reject_unpinned_or_unsafe_sources_and_paths(self):
        for key, value in [("path", "../file.safetensors"), ("path", "diffusion_models/a\\b.safetensors"),
                           ("source_url", self.asset["source_url"] + "&token=SECRET"),
                           ("source_url", "http://civitai.com/api/download/models/1?fileId=2"),
                           ("sha256", "invalid"), ("size", 0)]:
            with self.assertRaises(downloader.DownloadError):
                downloader.validate_asset(dict(self.asset, **{key: value}))
        with self.assertRaises(downloader.DownloadError):
            downloader.safe_https("https://cdn.example/file\nheader=unsafe")

    def test_account_auth_is_unredirected_and_http_downgrades_rejected(self):
        opener = mock.MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.geturl.return_value = "https://cdn.example/file?signature=private"
        with mock.patch.object(downloader.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(downloader.signed_url(self.asset["source_url"], "account-secret", 30), response.geturl.return_value)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer account-secret")
        redirected = downloader.HTTPSRedirect().redirect_request(request, None, 302, "Found", {}, response.geturl.return_value)
        self.assertNotIn("Authorization", dict(redirected.header_items()))
        with self.assertRaises(downloader.DownloadError):
            downloader.HTTPSRedirect().redirect_request(request, None, 302, "Found", {}, "http://cdn.example/file")

    def test_auth_failure_is_safe_and_permanent(self):
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://bad/?secret", 401, "PRIVATE", {}, None)
        with mock.patch.object(downloader.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(downloader.AuthenticationError, "CIVITAI_API_TOKEN") as raised:
                downloader.signed_url(self.asset["source_url"], "account-secret", 30)
        self.assertNotIn("PRIVATE", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_aria2_uses_private_stdin_parallel_resume_and_no_console_output(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(downloader.subprocess, "Popen") as popen:
            proc = popen.return_value
            proc.returncode = 0
            proc.poll.return_value = 0
            signed = "https://cdn.example/file?signature=private"
            downloader.transfer(signed, Path(temp) / "test.part", 16, 120)
            args, kwargs = popen.call_args
            self.assertNotIn(signed, " ".join(args[0]))
            self.assertIn("--continue=true", args[0])
            self.assertIn("--split=16", args[0])
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            self.assertNotIn("CIVITAI_API_TOKEN", kwargs["env"])
            self.assertNotIn("HF_TOKEN", kwargs["env"])
            self.assertEqual(proc.communicate.call_args.args, (signed + "\n  out=test.part\n",))

    def test_valid_file_is_sha_checked_and_does_not_redownload(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(downloader, "transfer") as transfer:
            path = Path(temp) / self.asset["path"]
            path.parent.mkdir()
            path.write_bytes(self.payload)
            downloader.download(self.asset, Path(temp), "secret", retries=1, connections=4, timeout=30)
            transfer.assert_not_called()
            path.write_bytes(self.payload[:-1] + b"X")
            self.assertFalse(downloader.checked(path, self.asset))  # Same size, wrong SHA.

    def test_interrupted_partial_resumes_and_complete_file_is_atomic(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(downloader, "signed_url", return_value="https://cdn.example/file"), \
                mock.patch.object(downloader.time, "sleep"):
            root = Path(temp)
            final = root / self.asset["path"]
            calls = []
            def transfer(url, partial, connections, timeout):
                self.assertFalse(final.exists())
                if not calls:
                    calls.append(1)
                    partial.write_bytes(self.payload[:20])
                    partial.with_name(partial.name + ".aria2").write_bytes(b"resume")
                    raise RuntimeError("PRIVATE signed URL account-secret")
                self.assertEqual(partial.read_bytes(), self.payload[:20])
                partial.write_bytes(self.payload)
                partial.with_name(partial.name + ".aria2").unlink()
            err = io.StringIO()
            with mock.patch.object(downloader, "transfer", side_effect=transfer), redirect_stderr(err):
                downloader.download(self.asset, root, "account-secret", retries=2, connections=4, timeout=30)
            self.assertEqual(final.read_bytes(), self.payload)
            self.assertFalse(final.with_name(final.name + ".part").exists())
            self.assertNotIn("PRIVATE", err.getvalue())
            self.assertNotIn("account-secret", err.getvalue())

    def test_corrupt_complete_download_never_becomes_model(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(downloader, "signed_url", return_value="https://cdn.example/file"):
            def transfer(url, partial, connections, timeout):
                partial.write_bytes(self.payload[:-1] + b"X")
            with mock.patch.object(downloader, "transfer", side_effect=transfer):
                with self.assertRaises(downloader.DownloadError):
                    downloader.download(self.asset, Path(temp), "secret", retries=1, connections=4, timeout=30)
            self.assertFalse((Path(temp) / self.asset["path"]).exists())


@unittest.skipUnless(entrypoint_tests.BASH and Path(entrypoint_tests.BASH).is_file(), "bash unavailable")
class EntrypointTests(unittest.TestCase):
    def test_one_preset_with_selected_model_and_missing_token_fails_closed(self):
        harness = entrypoint_tests.CharacterEntrypointTests()
        with tempfile.TemporaryDirectory(prefix=".entrypoint-test-", dir=ROOT) as temp:
            root, env = harness.make_install(temp)
            for profile in ("dasiwa-v2", "dasiwa-turbo-v2", "official"):
                env.update(H3_PROFILE="r2v", H3_R2V_MODEL=profile)
                result = harness.run_entrypoint(env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                stored = json.loads((root / "user/default/workflows" / profiles.WORKFLOW).read_text(encoding="utf-8"))
                self.assertEqual(stored, profiles.prepare(profile)[2])
                self.assertEqual(len(list((root / "user/default/workflows").glob("*.json"))), 1)
            env.update(H3_R2V_MODEL="dasiwa-v2", AUTO_MOSAIC_REQUIRED="0", CIVITAI_TOKEN="", CIVITAI_API_TOKEN="")
            result = harness.run_entrypoint(env)
            self.assertEqual(result.returncode, 70, result.stdout + result.stderr)
            self.assertNotIn("entrypoint contract passed", result.stdout)
            env["H3_R2V_MODEL"] = "typo"
            self.assertEqual(harness.run_entrypoint(env).returncode, 79)


if __name__ == "__main__":
    unittest.main()
