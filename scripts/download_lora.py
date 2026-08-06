#!/usr/bin/env python3
"""Download the private HMMotion LoRA at Pod startup without logging its token."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

DEFAULT_REPO_ID = "uwgm/nikke-civitai-backup"
DEFAULT_SOURCE_PATH = "hmmotion_minimax-h3_epoch12.safetensors"
DESTINATION_NAME = "hmmotion_minimax-h3_epoch12.safetensors"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Invalid H3_LORA_SOURCE_PATH: {value!r}")
    return path.as_posix()


def inspect_safetensors(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    if size < 10:
        raise RuntimeError(f"LoRA is too small to be a safetensors file: {size} bytes")
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        if header_size < 2 or header_size > min(size - 8, 256 * 1024 * 1024):
            raise RuntimeError(f"Invalid safetensors header size: {header_size}")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensors = {
        key: value
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }
    if not tensors:
        raise RuntimeError("LoRA safetensors file contains no tensors")
    data_size = size - 8 - header_size
    for key, tensor in tensors.items():
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise RuntimeError(f"Invalid safetensors data offsets for tensor {key!r}")
    return size, len(tensors)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    required = env_flag("H3_LORA_REQUIRED", True)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        message = "HF_TOKEN is required to download the private HMMotion LoRA"
        if required:
            raise RuntimeError(message)
        print(f"[lora] {message}; optional download skipped")
        return 0

    repo_id = os.environ.get("H3_LORA_REPO_ID", DEFAULT_REPO_ID).strip()
    source_path = validate_source_path(
        os.environ.get("H3_LORA_SOURCE_PATH", DEFAULT_SOURCE_PATH).strip()
    )
    requested_revision = os.environ.get("H3_LORA_REVISION", "main").strip()
    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    lora_dir = model_root / "loras"
    destination = lora_dir / DESTINATION_NAME
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    expected_sha = os.environ.get("H3_LORA_SHA256", "").strip().lower()
    if expected_sha and (len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha)):
        raise ValueError("H3_LORA_SHA256 must be a lowercase 64-character SHA256 digest")

    lora_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    last_error: Exception | None = None
    resolved_revision = requested_revision
    download_path: Path | None = None
    for attempt in range(1, retries + 1):
        try:
            info = api.model_info(repo_id, revision=requested_revision, token=token)
            resolved_revision = info.sha
            print(
                f"[lora] download attempt {attempt}/{retries}: "
                f"{repo_id}@{resolved_revision}:{source_path}"
            )
            download_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=source_path,
                    revision=resolved_revision,
                    token=token,
                    local_dir=lora_dir,
                )
            )
            break
        except Exception as error:  # huggingface_hub has several transport errors
            last_error = error
            if attempt < retries:
                delay = attempt * 5
                print(f"[lora] download failed; retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
    if download_path is None:
        raise RuntimeError(
            "HMMotion LoRA download failed. Confirm HF_TOKEN has read access to "
            f"https://huggingface.co/{repo_id}"
        ) from last_error

    if download_path.resolve() != destination.resolve():
        shutil.copyfile(download_path, destination)
    size, tensor_count = inspect_safetensors(destination)
    actual_sha = ""
    if expected_sha:
        actual_sha = sha256(destination)
        if actual_sha != expected_sha:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"HMMotion LoRA SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )

    record = {
        "repo_id": repo_id,
        "source_path": source_path,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "destination": str(destination),
        "size": size,
        "tensor_count": tensor_count,
    }
    if actual_sha:
        record["sha256"] = actual_sha
    record_path = Path(
        os.environ.get(
            "H3_LORA_RECORD_PATH",
            "/opt/ComfyUI/user/default/hmmotion_lora_source.json",
        )
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"[lora] ready: {destination.name}, {size} bytes, {tensor_count} tensors; "
        f"source revision {resolved_revision}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[lora] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
