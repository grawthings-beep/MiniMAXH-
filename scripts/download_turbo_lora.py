#!/usr/bin/env python3
"""Download and verify both selectable LightX2V MiniMax H3 Turbo LoRAs."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from download_lora import env_flag, inspect_safetensors, sha256


@dataclass(frozen=True)
class TurboAsset:
    key: str
    repo_id: str
    revision: str
    source_path: str
    destination_name: str
    expected_size: int
    expected_sha256: str


TURBO_ASSETS = (
    TurboAsset(
        key="8STEP",
        repo_id="Kutches/minmax",
        revision="29bca53f5e27ed855fc00e54519443387ddf8691",
        source_path=(
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        ),
        destination_name=(
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        ),
        expected_size=1_956_193_000,
        expected_sha256=(
            "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"
        ),
    ),
    TurboAsset(
        key="4STEP",
        repo_id="lightx2v/Minimax-h3-Turbo",
        revision="2f015e66b37c585cea9dc4ae6f1850ea8788e742",
        source_path=(
            "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
        ),
        destination_name=(
            "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
        ),
        expected_size=1_956_193_000,
        expected_sha256=(
            "c8168ebc17bbacc4296103dda2fec1ba85b24392fa08cf2bfbcef0cff0dc3cc8"
        ),
    ),
)


def validate_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Invalid Turbo LoRA source path: {value!r}")
    return path.as_posix()


def configured_asset(asset: TurboAsset) -> TurboAsset:
    prefix = f"H3_TURBO_{asset.key}"
    source_path = validate_source_path(
        os.environ.get(f"{prefix}_SOURCE_PATH", asset.source_path).strip()
    )
    expected_size = int(
        os.environ.get(f"{prefix}_SIZE", str(asset.expected_size))
    )
    expected_sha = os.environ.get(
        f"{prefix}_SHA256", asset.expected_sha256
    ).strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError(
            f"{prefix}_SHA256 must be a lowercase 64-character SHA256 digest"
        )
    return TurboAsset(
        key=asset.key,
        repo_id=os.environ.get(f"{prefix}_REPO_ID", asset.repo_id).strip(),
        revision=os.environ.get(f"{prefix}_REVISION", asset.revision).strip(),
        source_path=source_path,
        destination_name=asset.destination_name,
        expected_size=expected_size,
        expected_sha256=expected_sha,
    )


def verify(path: Path, asset: TurboAsset) -> tuple[int, int]:
    size, tensor_count = inspect_safetensors(path)
    if size != asset.expected_size:
        raise RuntimeError(
            f"Turbo {asset.key} LoRA size mismatch: "
            f"expected {asset.expected_size}, got {size}"
        )
    actual_sha = sha256(path)
    if actual_sha != asset.expected_sha256:
        raise RuntimeError(
            f"Turbo {asset.key} LoRA SHA256 mismatch: "
            f"expected {asset.expected_sha256}, got {actual_sha}"
        )
    return size, tensor_count


def download_asset(
    asset: TurboAsset,
    *,
    lora_dir: Path,
    retries: int,
    token: str | None,
) -> dict[str, object]:
    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = HfApi(token=token).model_info(
        asset.repo_id,
        revision=asset.revision,
        token=token,
    ).sha

    destination = lora_dir / asset.destination_name
    if destination.exists():
        try:
            size, tensor_count = verify(destination, asset)
            print(
                f"[turbo:{asset.key.lower()}] already ready: "
                f"{destination.name}, {size} bytes, {tensor_count} tensors"
            )
            return {
                "key": asset.key,
                "repo_id": asset.repo_id,
                "requested_revision": asset.revision,
                "resolved_revision": resolved_revision,
                "source_path": asset.source_path,
                "destination": str(destination),
                "size": size,
                "sha256": asset.expected_sha256,
                "tensor_count": tensor_count,
            }
        except Exception:
            destination.unlink(missing_ok=True)

    last_error: Exception | None = None
    download_path: Path | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[turbo:{asset.key.lower()}] download attempt {attempt}/{retries}: "
                f"{asset.repo_id}@{resolved_revision}:{asset.source_path}"
            )
            download_path = Path(
                hf_hub_download(
                    repo_id=asset.repo_id,
                    filename=asset.source_path,
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
                print(
                    f"[turbo:{asset.key.lower()}] download failed; "
                    f"retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
    if download_path is None:
        raise RuntimeError(f"LightX2V Turbo {asset.key} LoRA download failed") from last_error

    if download_path.resolve() != destination.resolve():
        shutil.copyfile(download_path, destination)
    try:
        size, tensor_count = verify(destination, asset)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    print(
        f"[turbo:{asset.key.lower()}] ready: {destination.name}, {size} bytes, "
        f"{tensor_count} tensors; SHA256 verified"
    )
    return {
        "key": asset.key,
        "repo_id": asset.repo_id,
        "requested_revision": asset.revision,
        "resolved_revision": resolved_revision,
        "source_path": asset.source_path,
        "destination": str(destination),
        "size": size,
        "sha256": asset.expected_sha256,
        "tensor_count": tensor_count,
    }


def main() -> int:
    if not env_flag("H3_TURBO_REQUIRED", True):
        print("[turbo] optional Turbo workflow disabled; skipping LoRA downloads")
        return 0

    if os.environ.get("H3_TURBO_SOURCE_PATH"):
        print(
            "[turbo] H3_TURBO_SOURCE_PATH is obsolete and ignored; both pinned 768p "
            "profiles are controlled by H3_TURBO_8STEP_SOURCE_PATH and "
            "H3_TURBO_4STEP_SOURCE_PATH"
        )

    if os.environ.get("H3_TURBO_REPO_ID") or os.environ.get("H3_TURBO_REVISION"):
        print(
            "[turbo] common H3_TURBO_REPO_ID/H3_TURBO_REVISION are obsolete and "
            "ignored; each profile is independently pinned"
        )
    assets = tuple(configured_asset(asset) for asset in TURBO_ASSETS)
    model_root = Path(os.environ.get("COMFYUI_MODEL_DIR", "/opt/ComfyUI/models"))
    lora_dir = model_root / "loras"
    retries = max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3")))
    token = os.environ.get("HF_TOKEN", "").strip() or None
    lora_dir.mkdir(parents=True, exist_ok=True)

    # Each Pod is ephemeral. Fetch the two independent Xet files concurrently so
    # adding the selectable 4-step profile does not serialize another ~2 GB wait.
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets)) as executor:
        futures = [
            executor.submit(
                download_asset,
                asset,
                lora_dir=lora_dir,
                retries=retries,
                token=token,
            )
            for asset in assets
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    record = {
        "profiles": sorted(results, key=lambda item: str(item["key"])),
    }
    record_path = Path(
        os.environ.get(
            "H3_TURBO_RECORD_PATH",
            "/opt/ComfyUI/user/default/minimax_h3_turbo_sources.json",
        )
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        "[turbo] both independently pinned selectable profiles are ready"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[turbo] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
