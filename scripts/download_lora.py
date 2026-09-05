#!/usr/bin/env python3
"""Download the private HMMotion V1 LoRA without logging its token."""

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
LORA_ID = "hmmotion_v1"
VALID_LORA_IDS = frozenset(
    {
        LORA_ID,
        "hmnsfw_aio_v2",
        "motion_booster_anime",
        "nsfw_anime_v04",
    }
)
DEFAULT_SIZE = 309_964_680
DEFAULT_SHA256 = "aa31d84116b689e840cd4e218c305a2995de448d84d48d35217efa70f6bb29bf"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def selected_lora_ids() -> set[str]:
    raw = os.environ.get("H3_LORA_SELECTION", "all").strip().lower()
    if raw == "all":
        return set(VALID_LORA_IDS)
    if raw in {"", "none"}:
        return set()
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = selected - VALID_LORA_IDS
    if unknown:
        raise ValueError(
            "H3_LORA_SELECTION contains unknown IDs: " + ", ".join(sorted(unknown))
        )
    return selected


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
    if LORA_ID not in selected_lora_ids():
        print(f"[lora:{LORA_ID}] not selected; skipping")
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    required = env_flag("H3_LORA_REQUIRED", True)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        message = "HF_TOKEN is required to download the private HMMotion LoRA"
        if required:
            raise RuntimeError(message)
        print(f"[lora:{LORA_ID}] {message}; optional download skipped")
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
    expected_size = int(os.environ.get("H3_LORA_SIZE", str(DEFAULT_SIZE)))
    expected_sha = os.environ.get("H3_LORA_SHA256", DEFAULT_SHA256).strip().lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
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
                f"[lora:{LORA_ID}] download attempt {attempt}/{retries}: "
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
                print(
                    f"[lora:{LORA_ID}] download failed; retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
    if download_path is None:
        raise RuntimeError(
            "HMMotion LoRA download failed. Confirm HF_TOKEN has read access to "
            f"https://huggingface.co/{repo_id}"
        ) from last_error

    if download_path.resolve() != destination.resolve():
        shutil.copyfile(download_path, destination)
    size, tensor_count = inspect_safetensors(destination)
    if size != expected_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"HMMotion V1 LoRA size mismatch: expected {expected_size}, got {size}"
        )
    actual_sha = sha256(destination)
    if actual_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"HMMotion V1 LoRA SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
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
        f"[lora:{LORA_ID}] ready: {destination.name}, {size} bytes, {tensor_count} tensors; "
        f"source revision {resolved_revision}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[lora:{LORA_ID}] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
