#!/usr/bin/env python3
"""Download and verify the public LightX2V MiniMax H3 Turbo LoRA."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

from download_lora import env_flag, inspect_safetensors, sha256


DEFAULT_REPO_ID = "lightx2v/Minimax-h3-Turbo"
DEFAULT_SOURCE_PATH = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
DEFAULT_REVISION = "05ef678438e84933c406131b59abbf86919b3aac"
DESTINATION_NAME = DEFAULT_SOURCE_PATH
EXPECTED_SIZE = 1_956_193_000
EXPECTED_SHA256 = "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"


def validate_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Invalid H3_TURBO_SOURCE_PATH: {value!r}")
    return path.as_posix()


def verify(path: Path, expected_size: int, expected_sha: str) -> tuple[int, int]:
    size, tensor_count = inspect_safetensors(path)
    if size != expected_size:
        raise RuntimeError(
            f"Turbo LoRA size mismatch: expected {expected_size}, got {size}"
        )
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Turbo LoRA SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return size, tensor_count


def main() -> int:
    if not env_flag("H3_TURBO_REQUIRED", True):
        print("[turbo] optional Turbo workflow disabled; skipping LoRA download")
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = os.environ.get("H3_TURBO_REPO_ID", DEFAULT_REPO_ID).strip()
    source_path = validate_source_path(
        os.environ.get("H3_TURBO_SOURCE_PATH", DEFAULT_SOURCE_PATH).strip()
    )
    requested_revision = os.environ.get(
        "H3_TURBO_REVISION", DEFAULT_REVISION
    ).strip()
    expected_size = int(os.environ.get("H3_TURBO_SIZE", str(EXPECTED_SIZE)))
    expected_sha = os.environ.get(
        "H3_TURBO_SHA256", EXPECTED_SHA256
    ).strip().lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("H3_TURBO_SHA256 must be a lowercase 64-character SHA256 digest")

    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    lora_dir = model_root / "loras"
    destination = lora_dir / DESTINATION_NAME
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    token = os.environ.get("HF_TOKEN", "").strip() or None
    lora_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            size, tensor_count = verify(destination, expected_size, expected_sha)
            print(
                f"[turbo] already ready: {destination.name}, {size} bytes, "
                f"{tensor_count} tensors"
            )
            return 0
        except Exception:
            destination.unlink(missing_ok=True)

    api = HfApi(token=token)
    last_error: Exception | None = None
    resolved_revision = requested_revision
    download_path: Path | None = None
    for attempt in range(1, retries + 1):
        try:
            info = api.model_info(repo_id, revision=requested_revision, token=token)
            resolved_revision = info.sha
            print(
                f"[turbo] download attempt {attempt}/{retries}: "
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
        except Exception as error:
            last_error = error
            if attempt < retries:
                delay = attempt * 5
                print(f"[turbo] download failed; retrying in {delay}s", file=sys.stderr)
                time.sleep(delay)
    if download_path is None:
        raise RuntimeError("LightX2V Turbo LoRA download failed") from last_error

    if download_path.resolve() != destination.resolve():
        shutil.copyfile(download_path, destination)
    try:
        size, tensor_count = verify(destination, expected_size, expected_sha)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    record = {
        "repo_id": repo_id,
        "source_path": source_path,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "destination": str(destination),
        "size": size,
        "sha256": expected_sha,
        "tensor_count": tensor_count,
    }
    record_path = Path(
        os.environ.get(
            "H3_TURBO_RECORD_PATH",
            "/opt/ComfyUI/user/default/minimax_h3_turbo_source.json",
        )
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"[turbo] ready: {destination.name}, {size} bytes, "
        f"{tensor_count} tensors; SHA256 verified"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[turbo] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
