#!/usr/bin/env python3
"""Merge pinned character R2V assets into the user's existing startup manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath


def merge_manifests(base: dict, character: dict) -> dict:
    # One HF download invocation has one repo/revision. Never silently repoint
    # user-selected models to a different revision or duplicate shared assets.
    for key in ("repo_id", "revision"):
        if base[key] != character[key]:
            raise ValueError(
                f"Character R2V requires the pinned {key}={character[key]}. "
                "Use MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_all.json "
                "or disable H3_CHARACTER_R2V; the custom manifest was not changed."
            )
    merged = copy.deepcopy(base)
    files = {}
    for item in [*base["files"], *character["files"]]:
        path = item["path"]
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path or ":" in path:
            raise ValueError("Manifest model paths must be relative and stay inside models.")
        if path in files and files[path] != item:
            raise ValueError(f"Conflicting model metadata: {path}")
        files[path] = copy.deepcopy(item)
    merged["files"] = list(files.values())
    merged["total_bytes"] = sum(item["size"] for item in merged["files"])
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.base.resolve(), args.character.resolve()):
        parser.error("Output must not overwrite an input manifest.")
    merged = merge_manifests(
        json.loads(args.base.read_text(encoding="utf-8")),
        json.loads(args.character.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"[character-r2v] {len(merged['files'])} unique models, {merged['total_bytes']} bytes")
