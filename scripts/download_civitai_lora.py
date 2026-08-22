#!/usr/bin/env python3
"""Download the pinned Civitai MiniMax H3 V2 LoRA without exposing its token."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from download_lora import env_flag, inspect_safetensors, selected_lora_ids, sha256


LORA_ID = "hmnsfw_aio_v2"
DESTINATION_NAME = "HMNSFW_AIO_V2.safetensors"
DEFAULT_SOURCE_URL = (
    "https://civitai.red/api/download/models/3206518?fileId=3088013"
)
MODEL_VERSION_ID = 3_206_518
FILE_ID = 3_088_013
EXPECTED_SIZE = 310_168_344
EXPECTED_SHA256 = (
    "608e4212f2788b6063330ff1196fc1f4b4228cfd9a413a63c198a09d7e4a61cb"
)


class PermanentDownloadError(RuntimeError):
    """A retry cannot fix this Civitai response."""


def validate_source_url(source_url: str) -> str:
    parsed_url = urllib.parse.urlsplit(source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in {"civitai.red", "civitai.com"}
        or not parsed_url.path.startswith("/api/download/models/")
    ):
        raise ValueError("H3_CIVITAI_LORA_URL must use HTTPS on civitai.red or civitai.com")
    if any(
        key.lower() == "token"
        for key, _ in urllib.parse.parse_qsl(parsed_url.query, keep_blank_values=True)
    ):
        raise ValueError("Do not put the Civitai token in H3_CIVITAI_LORA_URL")
    return source_url


def build_civitai_request(url: str, token: str, offset: int) -> urllib.request.Request:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "MiniMAXH-RunPod/0.6",
        },
        method="GET",
    )
    # urllib deliberately does not copy unredirected headers to the signed CDN URL.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return request


def open_civitai_download(url: str, token: str, offset: int, timeout: int):
    return urllib.request.urlopen(
        build_civitai_request(url, token, offset), timeout=timeout
    )


def download_once(
    url: str, token: str, partial: Path, timeout: int, expected_size: int
) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    try:
        response = open_civitai_download(url, token, offset, timeout)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise PermanentDownloadError(
                f"Civitai returned HTTP {error.code}; check CIVITAI_TOKEN and file access"
            ) from None
        if error.code == 416 and offset == expected_size:
            return
        raise RuntimeError(f"Civitai returned HTTP {error.code}") from None

    with response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def verify_download(path: Path, expected_size: int, expected_sha: str) -> tuple[int, int]:
    size, tensor_count = inspect_safetensors(path)
    if size != expected_size:
        raise RuntimeError(
            f"Civitai LoRA size mismatch: expected {expected_size}, got {size}"
        )
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Civitai LoRA SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return size, tensor_count


def main() -> int:
    if LORA_ID not in selected_lora_ids():
        print(f"[lora:{LORA_ID}] not selected; skipping")
        return 0

    required = env_flag("H3_LORA_REQUIRED", True)
    token = os.environ.get("CIVITAI_TOKEN", "").strip()
    if not token:
        message = "CIVITAI_TOKEN is required because the V2 creator requires login"
        if required:
            raise RuntimeError(message)
        print(f"[lora:{LORA_ID}] {message}; optional download skipped")
        return 0

    source_url = validate_source_url(
        os.environ.get("H3_CIVITAI_LORA_URL", DEFAULT_SOURCE_URL).strip()
    )
    expected_size = int(
        os.environ.get("H3_CIVITAI_LORA_SIZE", str(EXPECTED_SIZE))
    )
    expected_sha = os.environ.get(
        "H3_CIVITAI_LORA_SHA256", EXPECTED_SHA256
    ).strip().lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError(
            "H3_CIVITAI_LORA_SHA256 must be a lowercase 64-character SHA256 digest"
        )

    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    lora_dir = model_root / "loras"
    destination = lora_dir / DESTINATION_NAME
    partial = lora_dir / f"{DESTINATION_NAME}.part"
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    timeout = max(30, int(os.environ.get("CIVITAI_DOWNLOAD_TIMEOUT", "120")))
    lora_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            size, tensor_count = verify_download(
                destination, expected_size, expected_sha
            )
            print(
                f"[lora:{LORA_ID}] already ready: {destination.name}, "
                f"{size} bytes, {tensor_count} tensors"
            )
            return 0
        except Exception:
            destination.unlink(missing_ok=True)

    last_error: Exception | None = None
    size = tensor_count = 0
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[lora:{LORA_ID}] download attempt {attempt}/{retries}: "
                f"version {MODEL_VERSION_ID}, file {FILE_ID}"
            )
            download_once(source_url, token, partial, timeout, expected_size)
            size, tensor_count = verify_download(partial, expected_size, expected_sha)
            os.replace(partial, destination)
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
                delay = attempt * 5
                print(
                    f"[lora:{LORA_ID}] download interrupted at {partial_size} bytes; "
                    f"retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
    else:
        raise RuntimeError(
            "Civitai V2 LoRA download failed after integrity-checked retries"
        ) from last_error

    record = {
        "model_version_id": MODEL_VERSION_ID,
        "file_id": FILE_ID,
        "source_url": source_url,
        "destination": str(destination),
        "size": size,
        "tensor_count": tensor_count,
        "sha256": expected_sha,
    }
    record_path = Path(
        os.environ.get(
            "H3_CIVITAI_LORA_RECORD_PATH",
            "/opt/ComfyUI/user/default/hmnsfw_aio_v2_source.json",
        )
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"[lora:{LORA_ID}] ready: {destination.name}, {size} bytes, "
        f"{tensor_count} tensors; SHA256 verified"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[lora:{LORA_ID}] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
