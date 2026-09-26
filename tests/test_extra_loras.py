from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_extra_loras as extra


def tensor_bytes():
    header = json.dumps({"weight": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
    }}).encode()
    return len(header).to_bytes(8, "little") + header + b"\0" * 4


DATA = tensor_bytes()
DIGEST = hashlib.sha256(DATA).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data=DATA, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}


def version_metadata():
    return {
        "id": 12, "modelId": 34, "model": {"type": "LORA"},
        "files": [{"id": 56, "name": "test.safetensors", "type": "Model",
                   "primary": True, "hashes": {"SHA256": DIGEST.upper()},
                   "metadata": {"format": "SafeTensor", "fp": "bf16"}}],
    }


class ExtraLoRATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parse_comments_bom_duplicates_and_hf_file_viewer(self):
        urls = extra.parse_list("\ufeff# comment\n\n  https://example.com/a.safetensors  \n"
                                "https://example.com/a.safetensors\n"
                                "https://huggingface.co/owner/repo/blob/main/a.safetensors?download=true")
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[1], "https://huggingface.co/owner/repo/resolve/main/a.safetensors?download=true")

    def test_unsafe_urls_fail_without_echoing_secrets(self):
        for url in ("http://example.com/a", "file:///tmp/a", "https://user:secret@example.com/a",
                    "https://example.com:80/a", "https://example.com/a?token=secret",
                    "https://example.com/a?API_KEY=secret", "https://example.com/a b"):
            with self.subTest(url=url), self.assertRaises(extra.ListError) as caught:
                extra.parse_list(url)
            self.assertNotIn("secret", str(caught.exception).replace("secrets", ""))
            self.assertNotIn(url, str(caught.exception))

    def test_provider_tokens_are_host_scoped_and_not_redirected(self):
        os.environ.update(CIVITAI_TOKEN="civitai-secret", HF_TOKEN="hf-secret")
        for host, token in (("civitai.com", "civitai-secret"), ("civitai.red", "civitai-secret"),
                            ("huggingface.co", "hf-secret"), ("example.com", None),
                            ("civitai.com.example.com", None)):
            request = extra.build_request(f"https://{host}/a")
            self.assertEqual(request.get_header("Authorization"), f"Bearer {token}" if token else None)
            redirected = extra.HTTPSRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "https://cdn.example.com/file"
            )
            self.assertIsNone(redirected.get_header("Authorization"))
        del os.environ["CIVITAI_TOKEN"]
        os.environ["CIVITAI_API_TOKEN"] = "fallback"
        self.assertEqual(extra.build_request("https://civitai.red/a").get_header("Authorization"), "Bearer fallback")
        with self.assertRaises(extra.ListError):
            extra.HTTPSRedirectHandler().redirect_request(request, None, 302, "Found", {}, "http://cdn.example.com/a")

    def test_civitai_page_and_download_resolve_to_same_pinned_file(self):
        for url in ("https://civitai.com/models/34/test?modelVersionId=12",
                    "https://civitai.com/api/download/models/12?fileId=56",
                    "https://civitai.com/api/download/models/12?type=Model&format=SafeTensor&fp=bf16"):
            with mock.patch.object(extra, "read_json", return_value=version_metadata()):
                source = extra.resolve_source(url, 30)
            self.assertEqual(source.url, "https://civitai.com/api/download/models/12?fileId=56")
            self.assertEqual(source.expected_sha, DIGEST)
        with mock.patch.object(extra, "read_json", side_effect=[
            {"modelVersions": [{"id": 12}]}, version_metadata(),
        ]) as request:
            source = extra.resolve_source("https://civitai.red/models/34/test", 30)
        self.assertEqual(request.call_count, 2)
        self.assertIn("civitai.red/api/download/models/12?fileId=56", source.url)

    def test_civitai_file_filters_and_primary_do_not_pick_wrong_file(self):
        metadata = version_metadata()
        other = copy.deepcopy(metadata["files"][0])
        other.update(id=78, primary=False, name="other.safetensors")
        other["metadata"]["fp"] = "fp16"
        metadata["files"].append(other)
        with mock.patch.object(extra, "read_json", return_value=metadata):
            self.assertIn("fileId=56", extra.resolve_source("https://civitai.com/api/download/models/12", 30).url)
            self.assertIn("fileId=78", extra.resolve_source("https://civitai.com/api/download/models/12?fp=fp16", 30).url)
            with self.assertRaises(extra.ListError):
                extra.resolve_source("https://civitai.com/api/download/models/12?fileId=999", 30)
            metadata["files"][0]["primary"] = False
            with self.assertRaises(extra.ListError):
                extra.resolve_source("https://civitai.com/api/download/models/12", 30)

    def test_civitai_rejects_checkpoints_wrong_model_and_missing_hash(self):
        for mutate in (
            lambda data: data["model"].update(type="Checkpoint"),
            lambda data: data.update(modelId=999),
            lambda data: data["files"][0].update(hashes={}),
        ):
            metadata = version_metadata()
            mutate(metadata)
            with mock.patch.object(extra, "read_json", return_value=metadata):
                with self.assertRaises((extra.ListError, extra.DownloadError)):
                    extra.resolve_source("https://civitai.com/models/34/test?modelVersionId=12", 30)

    def test_remote_names_cannot_escape_and_same_names_do_not_collide(self):
        a = extra.Source("https://example.com/a", "../../evil\\bad.safetensors")
        b = extra.Source("https://example.com/b", a.name)
        self.assertNotIn("/", a.filename)
        self.assertNotIn("\\", a.filename)
        self.assertNotEqual(a.filename, b.filename)

    def test_install_atomic_verified_and_cached_without_network(self):
        source = extra.Source("https://example.com/test.safetensors", "test.safetensors", DIGEST)
        with mock.patch.object(extra, "open_url", return_value=Response()):
            result = extra.install(source, self.directory, 30)
        path = self.directory / "loras" / "extra" / source.filename
        self.assertEqual(path.read_bytes(), DATA)
        self.assertIn("ready:", result)
        with mock.patch.object(extra, "open_url", side_effect=AssertionError("unexpected download")):
            self.assertIn("already ready", extra.install(source, self.directory, 30))
        self.assertFalse(list(path.parent.glob("*.part")))
        receipt = json.loads(path.with_suffix(".safetensors.json").read_text())
        self.assertEqual(receipt["sha256"], DIGEST)
        self.assertNotIn("url", receipt)

    def test_generic_cache_detects_corruption_even_with_valid_header(self):
        source = extra.Source("https://example.com/test", "test")
        with mock.patch.object(extra, "open_url", side_effect=lambda *_: Response()):
            extra.install(source, self.directory, 30)
        path = self.directory / "loras" / "extra" / source.filename
        path.write_bytes(DATA[:-1] + b"\x01")
        with mock.patch.object(extra, "open_url", side_effect=lambda *_: Response()) as request:
            extra.install(source, self.directory, 30)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(path.read_bytes(), DATA)

    def test_bad_download_never_publishes_and_preserves_existing_file(self):
        cases = [Response(b"<html>login</html>", headers={"Content-Type": "text/html"}),
                 Response(DATA[:-1]), Response(DATA, headers={"Content-Length": "999999"}),
                 Response(DATA, status=206), Response(DATA[:-1] + b"\x01")]
        source = extra.Source("https://example.com/test", "test", DIGEST)
        directory = self.directory / "loras" / "extra"
        directory.mkdir(parents=True)
        destination = directory / source.filename
        for response in cases:
            with self.subTest(status=response.status):
                with mock.patch.object(extra, "open_url", return_value=response):
                    with self.assertRaises(extra.DownloadError):
                        extra.install(source, self.directory, 30)
                self.assertFalse(destination.exists())
                self.assertFalse(list(directory.glob("*.part")))
        destination.write_bytes(DATA)
        with mock.patch.object(extra, "open_url", return_value=Response(b"bad")):
            with self.assertRaises(extra.DownloadError):
                extra.install(source, self.directory, 30)
        self.assertEqual(destination.read_bytes(), DATA)

    def test_transient_retry_and_auth_failure_does_not_retry(self):
        operation = mock.Mock(side_effect=[extra.DownloadError("temporary"), "done"])
        with mock.patch.object(extra.time, "sleep"):
            self.assertEqual(extra.retry(operation, 3), "done")
        self.assertEqual(operation.call_count, 2)
        operation = mock.Mock(side_effect=extra.PermanentDownloadError("HTTP 403"))
        with self.assertRaises(extra.PermanentDownloadError):
            extra.retry(operation, 3)
        self.assertEqual(operation.call_count, 1)

    def test_http_errors_and_raw_exceptions_do_not_leak_urls(self):
        error = urllib.error.HTTPError("https://example.com/?token=secret", 403, "secret", {}, None)
        with mock.patch.object(extra.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(extra.PermanentDownloadError) as caught:
                extra.open_url("https://example.com/file", 30)
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("secret", extra.safe_error(ValueError("secret")))

    def test_remote_list_read_limit(self):
        with mock.patch.object(extra, "open_url", return_value=Response(b"12345")):
            with self.assertRaises(extra.ListError):
                extra.read_text("https://example.com/list", 30, limit=4)

    def test_main_remote_list_is_live_and_local_override_skips_network(self):
        os.environ["H3_EXTRA_LORA_LIST_URL"] = "https://raw.githubusercontent.com/owner/repo/main/list.txt"
        local = self.directory / "list.txt"
        local.write_text("# empty\n", encoding="utf-8")
        with mock.patch.object(extra, "read_text", return_value="# remote empty\n") as request:
            self.assertEqual(extra.main(["--check"]), 0)
            request.assert_called_once()
        with mock.patch.object(extra, "read_text", side_effect=AssertionError("unexpected network")):
            self.assertEqual(extra.main(["--list", str(local), "--check"]), 0)

    def test_required_optional_and_check_failure_status(self):
        local = self.directory / "list.txt"
        local.write_text("https://example.com/fail\nhttps://example.com/good\n", encoding="utf-8")
        for required, expected in ((False, 0), (True, 1)):
            with mock.patch.object(extra, "install", side_effect=[extra.PermanentDownloadError("denied"), "ready"]) as install:
                args = ["--list", str(local)] + (["--required"] if required else [])
                self.assertEqual(extra.main(args), expected)
                self.assertEqual(install.call_count, 2, "a failed URL must not prevent later downloads")
        local.write_text("https://example.com/a?token=secret", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(extra.main(["--list", str(local), "--check"]), 1)
        self.assertNotIn("token=secret", output.getvalue())

    def test_same_resolved_file_downloaded_once(self):
        local = self.directory / "list.txt"
        local.write_text("https://civitai.com/models/34/test?modelVersionId=12\n"
                         "https://civitai.com/api/download/models/12?fileId=56", encoding="utf-8")
        with mock.patch.object(extra, "read_json", return_value=version_metadata()):
            with mock.patch.object(extra, "install", return_value="ready") as install:
                self.assertEqual(extra.main(["--list", str(local), "--required"]), 0)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
