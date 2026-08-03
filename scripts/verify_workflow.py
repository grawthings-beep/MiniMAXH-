#!/usr/bin/env python3
"""Verify MiniMax H3 workflows, model sets, and native ComfyUI nodes."""

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


def graph_candidates(workflow: dict[str, object]) -> list[dict[str, object]]:
    graphs = [workflow]
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        graphs.append(subgraph)
    return graphs


def link_origin(link: object) -> int:
    return int(link["origin_id"] if isinstance(link, dict) else link[1])


def link_target(link: object) -> int:
    return int(link["target_id"] if isinstance(link, dict) else link[3])


def verify_easycache_wiring(workflow: dict[str, object]) -> None:
    graph = next(
        (
            candidate
            for candidate in graph_candidates(workflow)
            if any(node["type"] == "EasyCache" for node in candidate.get("nodes", []))
        ),
        None,
    )
    if graph is None:
        raise RuntimeError("EasyCache is missing from the fast workflow")
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    cache = next(node for node in nodes if node["type"] == "EasyCache")
    scheduler = next(node for node in nodes if node["type"] == "BasicScheduler")
    guider = next(node for node in nodes if node["type"] == "BasicGuider")
    if not any(
        link_origin(link) == int(unet["id"]) and link_target(link) == int(cache["id"])
        for link in links
    ):
        raise RuntimeError("UNETLoader is not connected to EasyCache")
    cache_targets = {
        link_target(link) for link in links if link_origin(link) == int(cache["id"])
    }
    if cache_targets != {int(scheduler["id"]), int(guider["id"])}:
        raise RuntimeError(
            "EasyCache must feed both BasicScheduler and BasicGuider; "
            f"targets={sorted(cache_targets)}"
        )
    if cache.get("widgets_values") != [0.2, 0.15, 0.95, True]:
        raise RuntimeError("EasyCache fast defaults have changed")


def verify_video_reference_wiring(workflow: dict[str, object]) -> None:
    graph = workflow
    nodes = graph["nodes"]
    links = graph["links"]
    load_video = next(node for node in nodes if node["type"] == "LoadVideo")
    components = next(node for node in nodes if node["type"] == "GetVideoComponents")
    r2v = next(node for node in nodes if node["type"] == "MiniMaxH3ReferenceToVideo")
    expected = {
        (int(load_video["id"]), int(components["id"]), "VIDEO"),
        (int(components["id"]), int(r2v["id"]), "IMAGE"),
        (int(components["id"]), int(r2v["id"]), "AUDIO"),
    }
    actual = {
        (
            link_origin(link),
            link_target(link),
            str(link[5] if not isinstance(link, dict) else link["type"]),
        )
        for link in links
    }
    if not expected <= actual:
        raise RuntimeError(f"Native video-reference wiring is incomplete: {sorted(expected - actual)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--mode", choices=("i2v", "r2v"), required=True)
    parser.add_argument("--expect-easycache", action="store_true")
    parser.add_argument("--require-video-reference", action="store_true")
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
    if args.mode == "i2v":
        if "MiniMaxH3ImageToVideo" not in node_types:
            raise RuntimeError("MiniMaxH3ImageToVideo is missing from the I2V workflow")
        if "MiniMaxH3ReferenceToVideo" in node_types or any("ref2va" in path for path in actual_models):
            raise RuntimeError("Reference-to-video assets are not allowed in an I2V workflow")
    else:
        if "MiniMaxH3ReferenceToVideo" not in node_types:
            raise RuntimeError("MiniMaxH3ReferenceToVideo is missing from the R2V workflow")
        if "MiniMaxH3ImageToVideo" in node_types or any("fl2va" in path for path in actual_models):
            raise RuntimeError("FL2VA assets are not allowed in an R2V workflow")
    if args.expect_easycache:
        verify_easycache_wiring(workflow)
    elif "EasyCache" in node_types:
        raise RuntimeError("EasyCache is enabled in a Quality workflow")
    if args.require_video_reference:
        verify_video_reference_wiring(workflow)

    if args.comfyui_root:
        verify_comfyui_nodes(args.comfyui_root, node_types, subgraph_ids)

    native_count = len(node_types - subgraph_ids)
    speed = "EasyCache Fast" if args.expect_easycache else "Quality"
    print(
        f"Verified {args.mode.upper()} {speed} workflow: "
        f"{len(actual_models)} models, {native_count} native node types."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
