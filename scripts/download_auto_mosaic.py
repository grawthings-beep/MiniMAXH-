#!/usr/bin/env python3
"""Securely resume, verify, and extract the pinned auto-mosaic model archive."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_MANIFEST = Path("/opt/minimax-h3/manifests/auto_mosaic.json")


class PermanentDownloadError(RuntimeError):
    """A retry cannot repair this download response."""


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_url(source_url: str) -> str:
    parsed = urllib.parse.urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"civitai.com", "www.civitai.com"}
        or not parsed.path.startswith("/api/download/models/")
    ):
        raise ValueError("Auto-mosaic URL must use the Civitai HTTPS model-download API")
    if any(
        key.lower() == "token"
        for key, _ in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    ):
        raise ValueError("Do not put CIVITAI_API_TOKEN in the auto-mosaic URL")
    return source_url


def build_request(url: str, token: str, offset: int) -> urllib.request.Request:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/zip,application/octet-stream",
            "User-Agent": "MiniMAXH-RunPod/0.8",
        },
        method="GET",
    )
    # urllib intentionally strips unredirected headers at the signed CDN redirect.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return request


def download_once(
    url: str, token: str, partial: Path, timeout: int, expected_size: int
) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    try:
        response = urllib.request.urlopen(build_request(url, token, offset), timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise PermanentDownloadError(
                f"Civitai returned HTTP {error.code}; check CIVITAI_API_TOKEN and model access"
            ) from None
        if error.code == 416 and offset == expected_size:
            return
        raise RuntimeError(f"Civitai returned HTTP {error.code}") from None
    with response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        with partial.open("ab" if append else "wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def pinned_member(archive: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    matches = [
        info for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name == basename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Auto-mosaic ZIP must contain exactly one {basename}, got {len(matches)}"
        )
    return matches[0]


def verify_archive(path: Path, expected_size: int, expected_sha: str) -> zipfile.ZipInfo:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Auto-mosaic archive size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Auto-mosaic archive SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Auto-mosaic ZIP CRC failed for {corrupt}")
        return pinned_member(archive, "ntd11_anime_nsfw_segm_v5.pt")


def extract_verified_member(archive_path: Path, member: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        info = pinned_member(archive, member)
        with archive.open(info) as source, partial.open("wb") as output:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    if partial.stat().st_size != info.file_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Extracted auto-mosaic model size does not match the ZIP member")
    os.replace(partial, destination)


def load_spec(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("Auto-mosaic manifest must contain exactly one archive")
    item = files[0]
    if item.get("provides") != ["auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"]:
        raise ValueError("Auto-mosaic manifest destination changed unexpectedly")
    if item.get("archive_member") != "ntd11_anime_nsfw_segm_v5.pt":
        raise ValueError("Auto-mosaic manifest ZIP member changed unexpectedly")
    return item


def main() -> int:
    if not env_flag("AUTO_MOSAIC_REQUIRED", True):
        print("[auto-mosaic] disabled; skipping")
        return 0
    token = os.environ.get("CIVITAI_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CIVITAI_API_TOKEN is required for the auto-mosaic model")

    manifest_path = Path(os.environ.get("AUTO_MOSAIC_MANIFEST", str(DEFAULT_MANIFEST)))
    item = load_spec(manifest_path)
    url = validate_source_url(str(item["url"]))
    expected_size = int(item["size_bytes"])
    expected_sha = str(item["sha256"]).lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("Auto-mosaic SHA256 must be a lowercase 64-character digest")

    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    archive_path = model_root / str(item["path"])
    destination = model_root / str(item["provides"][0])
    partial = archive_path.with_name(archive_path.name + ".part")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    timeout = max(30, int(os.environ.get("CIVITAI_DOWNLOAD_TIMEOUT", "120")))

    archive_info = None
    if archive_path.exists():
        try:
            archive_info = verify_archive(archive_path, expected_size, expected_sha)
        except Exception:
            archive_path.unlink(missing_ok=True)
    last_error: Exception | None = None
    if archive_info is None:
        for attempt in range(1, retries + 1):
            try:
                print(f"[auto-mosaic] download attempt {attempt}/{retries}: model 2266294")
                download_once(url, token, partial, timeout, expected_size)
                archive_info = verify_archive(partial, expected_size, expected_sha)
                os.replace(partial, archive_path)
                break
            except PermanentDownloadError:
                partial.unlink(missing_ok=True)
                raise
            except Exception as error:
                last_error = error
                partial_size = partial.stat().st_size if partial.exists() else 0
                if partial_size >= expected_size:
                    partial.unlink(missing_ok=True)
                    partial_size = 0
                if attempt < retries:
                    delay = attempt * 3
                    print(
                        f"[auto-mosaic] interrupted at {partial_size} bytes; retrying in {delay}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
        else:
            raise RuntimeError(
                "Auto-mosaic model download failed after integrity-checked retries"
            ) from last_error

    assert archive_info is not None
    if not destination.is_file() or destination.stat().st_size != archive_info.file_size:
        extract_verified_member(archive_path, str(item["archive_member"]), destination)
    if destination.stat().st_size != archive_info.file_size:
        raise RuntimeError("Auto-mosaic model extraction verification failed")
    print(
        f"[auto-mosaic] ready: {destination.name}, {destination.stat().st_size} bytes; "
        "archive size/SHA256 and member CRC verified"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[auto-mosaic] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
