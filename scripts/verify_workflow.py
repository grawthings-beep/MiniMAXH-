#!/usr/bin/env python3
"""Verify that the pinned I2V workflow is complete and uses native ComfyUI nodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRONTEND_BUILTINS = {"MarkdownNote"}


def all_nodes(workflow: dict[str, object]) -> list[dict[str, object]]:
    nodes = list(workflow.get("nodes", []))
    definitions = workflow.get("definitions", {})
    if isinstance(definitions, dict):
        for subgraph in definitions.get("subgraphs", []):
            nodes.extend(subgraph.get("nodes", []))
    return nodes


def workflow_models(nodes: list[dict[str, object]]) -> set[str]:
    models: set[str] = set()
    for node in nodes:
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            continue
        for model in properties.get("models", []):
            directory = str(model["directory"]).strip("/\\")
            name = str(model["name"])
            models.add(f"{directory}/{name}")
    return models


def source_has_node(source: str, node_type: str) -> bool:
    escaped = re.escape(node_type)
    patterns = (
        rf'node_id\s*=\s*["\']{escaped}["\']',
        rf'["\']{escaped}["\']\s*:',
        rf'class\s+{escaped}\b',
    )
    return any(re.search(pattern, source) for pattern in patterns)


def verify_comfyui_nodes(
    comfyui_root: Path, node_types: set[str], subgraph_ids: set[str]
) -> None:
    source_files = [comfyui_root / "nodes.py", *sorted((comfyui_root / "comfy_extras").glob("*.py"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    required_backend = node_types - subgraph_ids - FRONTEND_BUILTINS
    missing = sorted(node for node in required_backend if not source_has_node(source, node))
    if missing:
        raise RuntimeError(f"ComfyUI is missing workflow nodes: {', '.join(missing)}")

    requirements = (comfyui_root / "requirements.txt").read_text(encoding="utf-8")
    if FRONTEND_BUILTINS & node_types and "comfyui-frontend-package==" not in requirements:
        raise RuntimeError("ComfyUI frontend package is not pinned for frontend workflow nodes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    nodes = all_nodes(workflow)
    node_types = {str(node["type"]) for node in nodes}
    subgraph_ids = {
        str(subgraph["id"])
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
    }

    expected_models = {str(item["path"]) for item in manifest["files"]}
    actual_models = workflow_models(nodes)
    if actual_models != expected_models:
        missing = sorted(expected_models - actual_models)
        extra = sorted(actual_models - expected_models)
        raise RuntimeError(f"Workflow/model mismatch; missing={missing}, extra={extra}")
    if "MiniMaxH3ImageToVideo" not in node_types:
        raise RuntimeError("MiniMaxH3ImageToVideo is missing from the workflow")
    if "MiniMaxH3ReferenceToVideo" in node_types or any("ref2va" in path for path in actual_models):
        raise RuntimeError("Reference-to-video assets are not allowed in the I2V-only workflow")

    if args.comfyui_root:
        verify_comfyui_nodes(args.comfyui_root, node_types, subgraph_ids)

    native_count = len(node_types - subgraph_ids)
    print(f"Verified I2V workflow: {len(actual_models)} models, {native_count} native node types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
