#!/usr/bin/env python3
"""Install extra safetensors LoRAs from a local or remote one-URL-per-line list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from download_lora import env_flag, inspect_safetensors, sha256

DEFAULT_LIST = Path(__file__).resolve().parents[1] / "manifests" / "extra_loras.txt"
CIVITAI_HOSTS = {"civitai.com", "civitai.red", "www.civitai.com", "www.civitai.red"}
SECRET_KEYS = {"token", "access_token", "api_key", "apikey", "key", "authorization"}


class ListError(ValueError):
    """Configuration error whose message contains no source URL or token."""


class DownloadError(RuntimeError):
    """Safe-to-log download failure."""


class PermanentDownloadError(DownloadError):
    """Retrying cannot fix this response."""


def validate_transport(url: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        valid = (
            parsed.scheme == "https" and parsed.hostname
            and parsed.port in {None, 443} and not parsed.username
            and not parsed.password and not re.search(r"[\s\\\x00-\x1f]", url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ListError("Use an HTTPS URL without credentials or a custom port")
    return parsed


def normalize_url(url: str) -> str:
    parsed = validate_transport(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if any(key.lower() in SECRET_KEYS for key, _ in query):
        raise ListError("Remove tokens from URLs; use CIVITAI_TOKEN or HF_TOKEN secrets")
    path = parsed.path
    # Accept the link copied from Hugging Face's file viewer as well as resolve URLs.
    if parsed.hostname == "huggingface.co":
        path = re.sub(r"^(/[^/]+/[^/]+)/blob/", r"\1/resolve/", path)
    return urllib.parse.urlunsplit(("https", parsed.hostname, path, parsed.query, ""))


def parse_list(contents: str) -> list[str]:
    urls = []
    for line_number, raw in enumerate(contents.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            url = normalize_url(line)
        except ListError as error:
            raise ListError(f"Line {line_number}: {error}") from None
        if url not in urls:
            urls.append(url)
    return urls


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_transport(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_request(url: str) -> urllib.request.Request:
    parsed = validate_transport(url)
    request = urllib.request.Request(url, headers={
        "User-Agent": "MiniMAXH-ExtraLoRA/1.0", "Accept-Encoding": "identity",
    })
    token = ""
    if parsed.hostname in CIVITAI_HOSTS:
        token = os.environ.get("CIVITAI_TOKEN", "").strip() or os.environ.get(
            "CIVITAI_API_TOKEN", ""
        ).strip()
    elif parsed.hostname == "huggingface.co":
        token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        # Do not forward account secrets to redirected CDN hosts.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def open_url(url: str, timeout: int):
    try:
        return urllib.request.build_opener(HTTPSRedirectHandler()).open(
            build_request(url), timeout=timeout
        )
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        if code in {401, 403}:
            raise PermanentDownloadError(
                f"HTTP {code}: check CIVITAI_TOKEN / HF_TOKEN and file access"
            ) from None
        if code in {400, 404, 410}:
            raise PermanentDownloadError(f"HTTP {code}: check the URL and file availability") from None
        raise DownloadError(f"HTTP {code}") from None
    except (urllib.error.URLError, OSError, ValueError):
        raise DownloadError("Connection failed; check network and credentials") from None


def read_text(url: str, timeout: int, limit: int = 1024 * 1024) -> str:
    with open_url(url, timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ListError("URL list or metadata exceeds the size limit")
    return data.decode("utf-8-sig")


@dataclass(frozen=True)
class Source:
    url: str
    name: str
    expected_sha: str = ""

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]

    @property
    def filename(self) -> str:
        # Names from remote metadata must never become paths.
        stem = re.sub(r"[^\w.-]", "_", self.name.removesuffix(".safetensors"))
        stem = stem.strip("._")[:100] or "lora"
        return f"{stem}__{self.identity}.safetensors"


def read_json(url: str, timeout: int) -> dict:
    data = json.loads(read_text(url, timeout, 16 * 1024 * 1024))
    if not isinstance(data, dict):
        raise DownloadError("Invalid Civitai metadata")
    return data


def resolve_source(url: str, timeout: int) -> Source:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname not in CIVITAI_HOSTS:
        name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        return Source(url, name if name.lower().endswith(".safetensors") else "lora")

    host = parsed.hostname.removeprefix("www.")
    query = dict(urllib.parse.parse_qsl(parsed.query))
    download = re.fullmatch(r"/api/download/models/(\d+)/?", parsed.path)
    page = re.fullmatch(r"/models/(\d+)(?:/[^/]*)?/?", parsed.path)
    if download:
        version_id = download[1]
    elif page:
        version_id = query.get("modelVersionId", "")
        if not version_id:
            model = read_json(f"https://{host}/api/v1/models/{page[1]}", timeout)
            versions = model.get("modelVersions", [])
            if not versions:
                raise PermanentDownloadError("Civitai model has no downloadable versions")
            version_id = str(versions[0]["id"])
    else:
        raise ListError("Use a Civitai model page or /api/download/models/<version> URL")
    if not version_id.isdigit():
        raise ListError("Civitai modelVersionId must be numeric")
    version = read_json(f"https://{host}/api/v1/model-versions/{version_id}", timeout)
    if page and str(version.get("modelId")) != page[1]:
        raise ListError("Civitai modelVersionId does not belong to this model page")
    model_type = version.get("model", {}).get("type", "")
    if model_type and model_type.lower() not in {"lora", "locon"}:
        raise ListError("Civitai resource must be a LoRA or LoCon")
    files = [item for item in version.get("files", [])
             if item.get("type") == "Model"
             and item.get("name", "").lower().endswith(".safetensors")]
    for key in ("fileId", "type", "format", "size", "fp"):
        if key not in query:
            continue
        def matches(item):
            value = item.get("id") if key == "fileId" else (
                item.get("type") if key == "type" else item.get("metadata", {}).get(key)
            )
            return str(value).lower() == query[key].lower()
        files = [item for item in files if matches(item)]
    if len(files) > 1:
        primary = [item for item in files if item.get("primary")]
        if len(primary) == 1:
            files = primary
    if len(files) != 1:
        raise ListError("No unique safetensors LoRA matches; copy a download URL with fileId")
    selected = files[0]
    file_id = str(selected["id"])
    if not file_id.isdigit():
        raise DownloadError("Invalid Civitai file ID")
    digest = selected.get("hashes", {}).get("SHA256", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DownloadError("Civitai did not provide a valid SHA256 for this file")
    # Pin the selected file, including when a model page was pasted.
    return Source(
        f"https://{host}/api/download/models/{version_id}?fileId={file_id}",
        selected["name"], digest,
    )


def verify(path: Path, expected_sha: str = "") -> dict:
    size, tensor_count = inspect_safetensors(path)
    digest = sha256(path)
    if expected_sha and digest != expected_sha:
        raise DownloadError("SHA256 mismatch")
    return {"size": size, "tensor_count": tensor_count, "sha256": digest}


def atomic_record(path: Path, record: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".receipt-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install(source: Source, model_dir: Path, timeout: int) -> str:
    lora_dir = model_dir / "loras"
    directory = lora_dir / "extra"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.resolve().is_relative_to(lora_dir.resolve()):
        raise ListError("The extra LoRA directory must be inside models/loras")
    destination = directory / source.filename
    receipt_path = directory / f"{source.filename}.json"
    if destination.is_symlink() or receipt_path.is_symlink():
        raise ListError("Extra LoRA files must not be symbolic links")
    if destination.exists() and receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            digest = receipt["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid receipt")
            checked = verify(destination, source.expected_sha or digest)
            if receipt["source_id"] == source.identity and checked["sha256"] == digest:
                return f"already ready: extra/{source.filename}"
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, AttributeError):
            pass
    # A temporary suffix keeps incomplete files out of ComfyUI's LoRA selector.
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=directory)
    partial = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle, open_url(source.url, timeout) as response:
            if response.status != 200:
                raise DownloadError("Expected a complete HTTP 200 response")
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise PermanentDownloadError("Server returned a page/error instead of a LoRA file")
            length = response.headers.get("Content-Length")
            expected_length = int(length) if length is not None else None
            received = 0
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                received += len(chunk)
            if expected_length is not None and received != expected_length:
                raise DownloadError("Incomplete download (Content-Length mismatch)")
        try:
            record = verify(partial, source.expected_sha)
        except Exception:
            raise DownloadError("Invalid/truncated safetensors file or SHA256 mismatch") from None
        os.replace(partial, destination)
        record["source_id"] = source.identity
        atomic_record(receipt_path, record)
        return f"ready: extra/{source.filename} ({record['size']} bytes)"
    finally:
        partial.unlink(missing_ok=True)


def retry(operation, attempts: int):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (ListError, PermanentDownloadError):
            raise
        except Exception:
            if attempt == attempts:
                raise
            print(f"[extra-lora] interrupted; retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(min(attempt * 5, 15))


def safe_error(error: Exception) -> str:
    # urllib and JSON errors can contain signed URLs, tokens or response bodies.
    if isinstance(error, (ListError, DownloadError)):
        return str(error)
    return f"{type(error).__name__}: download/list processing failed (details withheld)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--list", type=Path, help="Local URL list; overrides the remote list")
    inputs.add_argument("--list-url", help="HTTPS URL of the plain-text URL list")
    parser.add_argument("--check", action="store_true", help="Validate list syntax without downloading models")
    parser.add_argument("--required", action="store_true", help="Return an error if any extra LoRA fails")
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get(
        "COMFYUI_MODEL_DIR", str(Path(os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI")) / "models")
    )))
    args = parser.parse_args(argv)
    required = args.required or args.check or env_flag("H3_EXTRA_LORA_REQUIRED", False)
    try:
        timeout = max(1, int(os.environ.get("EXTRA_LORA_DOWNLOAD_TIMEOUT", "120")))
        attempts = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
        list_url = "" if args.list else (
            args.list_url or os.environ.get("H3_EXTRA_LORA_LIST_URL", "").strip()
        )
        if list_url:
            list_url = normalize_url(list_url)
            contents = retry(lambda: read_text(list_url, timeout), attempts)
        else:
            path = args.list or Path(os.environ.get("H3_EXTRA_LORA_LIST", str(DEFAULT_LIST)))
            contents = path.read_text(encoding="utf-8-sig")
        urls = parse_list(contents)
    except Exception as error:
        print(f"[extra-lora] ERROR: {safe_error(error)}", file=sys.stderr)
        return 1 if required else 0
    print(f"[extra-lora] {len(urls)} unique URL(s) in list", flush=True)
    if args.check:
        return 0
    failures = 0
    installed = set()
    for index, url in enumerate(urls, 1):
        try:
            print(f"[extra-lora:{index}] resolving/downloading", flush=True)
            source = retry(lambda: resolve_source(url, timeout), attempts)
            if source.identity in installed:
                continue
            result = retry(lambda: install(source, args.model_dir, timeout), attempts)
            installed.add(source.identity)
            print(f"[extra-lora:{index}] {result}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[extra-lora:{index}] ERROR: {safe_error(error)}", file=sys.stderr)
    print(f"[extra-lora] complete: {len(installed)} ready, {failures} failed", flush=True)
    if failures and not required:
        print("[extra-lora] optional failures; continuing ComfyUI startup", file=sys.stderr)
    return 1 if failures and required else 0


if __name__ == "__main__":
    raise SystemExit(main())
