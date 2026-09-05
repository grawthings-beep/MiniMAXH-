#!/usr/bin/env python3
"""Download the selectable MiniMax H3 anime motion/style LoRAs."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from download_civitai_lora import (
    PermanentDownloadError,
    download_once,
    validate_source_url,
    verify_download,
)
from download_lora import inspect_safetensors, selected_lora_ids, sha256


MOTION_LORA_ID = "motion_booster_anime"
MOTION_DESTINATION = "H3_Motion_Booster_anime.safetensors"
MOTION_VERSION_ID = 3_299_686
MOTION_DEFAULT_URL = "https://civitai.com/api/download/models/3299686"
MOTION_METADATA_URL = "https://civitai.com/api/v1/model-versions/3299686"

STYLE_LORA_ID = "nsfw_anime_v04"
STYLE_DESTINATION = "NSFW_ANIME_V7_H3-step00019500.safetensors"
STYLE_REPO_ID = "Kutches/minmax"
STYLE_REVISION = "29bca53f5e27ed855fc00e54519443387ddf8691"
STYLE_SOURCE_PATH = STYLE_DESTINATION
STYLE_SIZE = 596_450_480
STYLE_SHA256 = "c69a8e719b6784a8e475004cd47d34d1ddefbb5daa2d7670632cd3b459490b8d"


def _valid_sha256(value: str, *, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must be a lowercase 64-character SHA256 digest")
    return digest


def validate_metadata_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    expected_path = f"/api/v1/model-versions/{MOTION_VERSION_ID}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "civitai.com"
        or parsed.path.rstrip("/") != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "H3_ANIME_MOTION_METADATA_URL must be the HTTPS Civitai API URL "
            f"for version {MOTION_VERSION_ID}"
        )
    return url


def _json_request(url: str, token: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        validate_metadata_url(url),
        headers={
            "Accept": "application/json",
            "User-Agent": "MiniMAXH-RunPod/0.7",
        },
        method="GET",
    )
    # Do not copy the account token if Civitai redirects metadata elsewhere.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Civitai version metadata was not a JSON object")
    return payload


def motion_file_metadata(
    payload: dict[str, Any], expected_name: str
) -> tuple[int, str]:
    files = payload.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Civitai version metadata contains no files list")
    exact = [item for item in files if item.get("name") == expected_name]
    if len(exact) != 1:
        found = ", ".join(str(item.get("name", "?")) for item in files)
        raise RuntimeError(
            f"Civitai version {MOTION_VERSION_ID} does not expose exactly one "
            f"{expected_name!r} file (found: {found})"
        )
    item = exact[0]
    file_id = int(item.get("id", 0))
    hashes = item.get("hashes")
    digest = hashes.get("SHA256") if isinstance(hashes, dict) else None
    if file_id <= 0 or not isinstance(digest, str):
        raise RuntimeError("Civitai motion LoRA metadata is missing file ID or SHA256")
    return file_id, _valid_sha256(digest, name="Civitai motion LoRA SHA256")


def _motion_download_url(base_url: str, file_id: int) -> str:
    validated = validate_source_url(base_url)
    parts = urllib.parse.urlsplit(validated)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query["fileId"] = str(file_id)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), "")
    )


def download_motion(
    *, model_root: Path, retries: int, timeout: int, token: str
) -> dict[str, Any]:
    expected_name = os.environ.get(
        "H3_ANIME_MOTION_FILENAME", MOTION_DESTINATION
    ).strip()
    if PurePosixPath(expected_name).name != expected_name:
        raise ValueError("H3_ANIME_MOTION_FILENAME must be a plain filename")
    payload = _json_request(
        os.environ.get("H3_ANIME_MOTION_METADATA_URL", MOTION_METADATA_URL).strip(),
        token,
        timeout,
    )
    file_id, metadata_sha = motion_file_metadata(payload, expected_name)
    expected_sha = _valid_sha256(
        os.environ.get("H3_ANIME_MOTION_SHA256", metadata_sha),
        name="H3_ANIME_MOTION_SHA256",
    )
    if expected_sha != metadata_sha:
        raise RuntimeError(
            "H3_ANIME_MOTION_SHA256 disagrees with Civitai version metadata"
        )
    source_url = _motion_download_url(
        os.environ.get("H3_ANIME_MOTION_URL", MOTION_DEFAULT_URL).strip(),
        file_id,
    )
    lora_dir = model_root / "loras"
    destination = lora_dir / MOTION_DESTINATION
    partial = lora_dir / f"{MOTION_DESTINATION}.part"
    lora_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            size, tensor_count = verify_download(destination, None, expected_sha)
            return {
                "id": MOTION_LORA_ID,
                "version_id": MOTION_VERSION_ID,
                "file_id": file_id,
                "destination": str(destination),
                "size": size,
                "sha256": expected_sha,
                "tensor_count": tensor_count,
            }
        except Exception:
            destination.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[lora:{MOTION_LORA_ID}] download attempt {attempt}/{retries}: "
                f"version {MOTION_VERSION_ID}, file {file_id}"
            )
            download_once(source_url, token, partial, timeout, None)
            try:
                size, tensor_count = verify_download(partial, None, expected_sha)
            except Exception:
                # A completed file with the wrong digest cannot be repaired by Range.
                partial.unlink(missing_ok=True)
                raise
            os.replace(partial, destination)
            break
        except PermanentDownloadError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as error:
            last_error = error
            if attempt < retries:
                delay = attempt * 5
                print(
                    f"[lora:{MOTION_LORA_ID}] interrupted; retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
    else:
        raise RuntimeError("Anime motion LoRA download failed") from last_error

    print(
        f"[lora:{MOTION_LORA_ID}] ready: {destination.name}, {size} bytes, "
        f"{tensor_count} tensors; SHA256 verified"
    )
    return {
        "id": MOTION_LORA_ID,
        "version_id": MOTION_VERSION_ID,
        "file_id": file_id,
        "destination": str(destination),
        "size": size,
        "sha256": expected_sha,
        "tensor_count": tensor_count,
    }


def download_style(
    *, model_root: Path, retries: int, token: str | None
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = os.environ.get("H3_ANIME_STYLE_REPO_ID", STYLE_REPO_ID).strip()
    revision = os.environ.get("H3_ANIME_STYLE_REVISION", STYLE_REVISION).strip()
    source_path = os.environ.get(
        "H3_ANIME_STYLE_SOURCE_PATH", STYLE_SOURCE_PATH
    ).strip()
    path = PurePosixPath(source_path)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError("Invalid H3_ANIME_STYLE_SOURCE_PATH")
    expected_size = int(os.environ.get("H3_ANIME_STYLE_SIZE", str(STYLE_SIZE)))
    expected_sha = _valid_sha256(
        os.environ.get("H3_ANIME_STYLE_SHA256", STYLE_SHA256),
        name="H3_ANIME_STYLE_SHA256",
    )
    lora_dir = model_root / "loras"
    destination = lora_dir / STYLE_DESTINATION
    lora_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            size, tensor_count = inspect_safetensors(destination)
            if size != expected_size or sha256(destination) != expected_sha:
                raise RuntimeError("cached anime style LoRA failed verification")
            return {
                "id": STYLE_LORA_ID,
                "repo_id": repo_id,
                "resolved_revision": revision,
                "destination": str(destination),
                "size": size,
                "sha256": expected_sha,
                "tensor_count": tensor_count,
            }
        except Exception:
            destination.unlink(missing_ok=True)

    api = HfApi(token=token)
    resolved_revision = api.model_info(
        repo_id, revision=revision, token=token
    ).sha
    last_error: Exception | None = None
    download_path: Path | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[lora:{STYLE_LORA_ID}] download attempt {attempt}/{retries}: "
                f"{repo_id}@{resolved_revision}:{path.as_posix()}"
            )
            download_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=path.as_posix(),
                    revision=resolved_revision,
                    token=token,
                    local_dir=lora_dir,
                )
            )
            break
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(attempt * 5)
    if download_path is None:
        raise RuntimeError("Anime style LoRA download failed") from last_error
    if download_path.resolve() != destination.resolve():
        shutil.copyfile(download_path, destination)
    size, tensor_count = inspect_safetensors(destination)
    actual_sha = sha256(destination)
    if size != expected_size or actual_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Anime style LoRA integrity mismatch: "
            f"expected {expected_size}/{expected_sha}, got {size}/{actual_sha}"
        )
    print(
        f"[lora:{STYLE_LORA_ID}] ready: {destination.name}, {size} bytes, "
        f"{tensor_count} tensors; SHA256 verified"
    )
    return {
        "id": STYLE_LORA_ID,
        "repo_id": repo_id,
        "resolved_revision": resolved_revision,
        "destination": str(destination),
        "size": size,
        "sha256": expected_sha,
        "tensor_count": tensor_count,
    }


def main() -> int:
    selected = selected_lora_ids()
    wanted = selected & {MOTION_LORA_ID, STYLE_LORA_ID}
    if not wanted:
        print("[anime-loras] no anime LoRA selected; skipping")
        return 0
    token = os.environ.get("CIVITAI_TOKEN", "").strip()
    if MOTION_LORA_ID in wanted and not token:
        raise RuntimeError(
            "CIVITAI_TOKEN is required for the selected anime motion LoRA"
        )
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    timeout = max(30, int(os.environ.get("CIVITAI_DOWNLOAD_TIMEOUT", "120")))

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(wanted)) as executor:
        if MOTION_LORA_ID in wanted:
            jobs.append(
                executor.submit(
                    download_motion,
                    model_root=model_root,
                    retries=retries,
                    timeout=timeout,
                    token=token,
                )
            )
        if STYLE_LORA_ID in wanted:
            jobs.append(
                executor.submit(
                    download_style,
                    model_root=model_root,
                    retries=retries,
                    token=hf_token,
                )
            )
        records = [future.result() for future in concurrent.futures.as_completed(jobs)]

    record_path = Path(
        os.environ.get(
            "H3_ANIME_LORA_RECORD_PATH",
            "/opt/ComfyUI/user/default/minimax_h3_anime_lora_sources.json",
        )
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(sorted(records, key=lambda item: str(item["id"])), indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[anime-loras] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
