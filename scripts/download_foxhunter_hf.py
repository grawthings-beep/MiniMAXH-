#!/usr/bin/env python3
"""Fetch FoxHunter's private HF LoRA without putting HF_TOKEN in logs or URLs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

from download_lora import inspect_safetensors
from verify_models import verify_one


def validate_asset(asset: dict) -> None:
    path = PurePosixPath(asset["path"])
    source = PurePosixPath(asset["source_path"])
    if (len(path.parts) != 3 or path.parts[:2] != ("loras", "MINIMAX")
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", path.name)):
        raise ValueError("Private HF asset must be a LoRA in loras/MINIMAX")
    if (len(source.parts) != 1 or source.name != path.name
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", source.name)):
        raise ValueError("Private HF source filename does not match the LoRA destination")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", asset["source_repo_id"]):
        raise ValueError("Invalid private HF repository ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", asset["source_revision"]):
        raise ValueError("Invalid private HF revision")
    if int(asset["size"]) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
        raise ValueError("Private HF LoRA needs a pinned size and SHA256")


def checked(root: Path, asset: dict) -> bool:
    path = root / asset["path"]
    if verify_one(root, asset, "sha256")[1] != "ok":
        return False
    try:
        inspect_safetensors(path)
    except (ValueError, RuntimeError, UnicodeError):
        return False
    return True


def download(asset: dict, root: Path, token: str, retries: int) -> None:
    validate_asset(asset)
    destination = root / asset["path"]
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Private HF LoRA destination escapes the model directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError("Private HF LoRA download paths must not be symlinks")
    if checked(root, asset):
        print(f"[foxhunter-hf] {destination.name} already verified", flush=True)
        return
    if not token or any(ord(c) < 32 for c in token):
        raise ValueError("HF_TOKEN with access to uwgm/illustrious is required")

    from huggingface_hub import HfApi, hf_hub_download

    for attempt in range(1, retries + 1):
        try:
            revision = HfApi(token=token).model_info(
                asset["source_repo_id"], revision=asset["source_revision"], token=token
            ).sha
            print(f"[foxhunter-hf] {destination.name}: attempt {attempt}/{retries}, revision {revision}", flush=True)
            cached = hf_hub_download(
                repo_id=asset["source_repo_id"], filename=asset["source_path"],
                revision=revision, token=token,
            )
            shutil.copyfile(cached, partial)
            if not checked(root, dict(asset, path=str(PurePosixPath(asset["path"]).with_name(partial.name)))):
                raise ValueError("Private HF LoRA size, SHA256 or safetensors verification failed")
            os.replace(partial, destination)
            print(f"[foxhunter-hf] {destination.name} ready; size and SHA256 verified", flush=True)
            return
        except Exception:
            # HF exceptions can include signed URLs or account details. Never print them.
            if attempt == retries:
                raise RuntimeError(
                    "Private HF LoRA download failed; check HF_TOKEN repository access "
                    "and that the supplied file matches the pinned SHA256"
                ) from None
            time.sleep(3 * attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = [item for item in manifest["files"] if item.get("auth") == "hf"]
    if len(assets) != 1:
        raise ValueError("FoxHunter requires exactly one private HF LoRA")
    download(assets[0], args.root, os.environ.get("HF_TOKEN", "").strip(),
             max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3"))))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[foxhunter-hf] ERROR: private LoRA setup failed; ComfyUI will not start", file=sys.stderr)
        raise SystemExit(1) from None
