#!/usr/bin/env python3
"""Verify exact MiniMax H3 model files from a pinned manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHUNK_SIZE = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_one(root: Path, item: dict[str, object], mode: str) -> tuple[str, str]:
    relative = str(item["path"])
    path = root / relative
    if not path.is_file():
        return relative, "missing"
    actual_size = path.stat().st_size
    expected_size = int(item["size"])
    if actual_size != expected_size:
        return relative, f"size mismatch: {actual_size} != {expected_size}"
    if mode == "sha256":
        actual_hash = sha256_file(path)
        expected_hash = str(item["sha256"])
        if actual_hash != expected_hash:
            return relative, f"sha256 mismatch: {actual_hash} != {expected_hash}"
    return relative, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("size", "sha256"), default="sha256")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--remove-invalid",
        action="store_true",
        help="Remove only known manifest files that fail size/hash validation.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest["files"]
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(verify_one, args.root, item, args.mode): item for item in files
        }
        for future in as_completed(futures):
            relative, status = future.result()
            results[relative] = status

    failed = False
    for item in files:
        relative = str(item["path"])
        status = results[relative]
        marker = "OK" if status == "ok" else "FAIL"
        print(f"[{marker}] {relative}: {status}")
        if status != "ok":
            failed = True
            if args.remove_invalid and status != "missing":
                target = args.root / relative
                target.unlink(missing_ok=True)
                print(f"[REMOVE] {relative}")

    if failed:
        print("Model verification failed.", file=sys.stderr)
        return 1
    print(f"Verified {len(files)} files using {args.mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
