#!/usr/bin/env python3
"""Build the mixed-reference R2V and native EasyCache workflow variants."""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path
from typing import Any


EASYCACHE_VALUES = [0.2, 0.15, 0.95, True]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def next_numeric_id(items: list[dict[str, Any]]) -> int:
    return max((int(item["id"]) for item in items if isinstance(item.get("id"), int)), default=0) + 1


def shift_orders(nodes: list[dict[str, Any]], first_order: int, amount: int) -> None:
    for node in nodes:
        if int(node.get("order", -1)) >= first_order:
            node["order"] = int(node["order"]) + amount


def prepare_r2v(upstream: dict[str, Any]) -> dict[str, Any]:
    """Add native video loading/extraction to the official image-reference template."""
    workflow = copy.deepcopy(upstream)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{workflow["id"]}:mixed-reference'))
    nodes = workflow["nodes"]
    links = workflow["links"]
    r2v = next(node for node in nodes if node["type"] == "MiniMaxH3ReferenceToVideo")

    load_video_id = next_numeric_id(nodes)
    components_id = load_video_id + 1
    first_link_id = max(int(link[0]) for link in links) + 1
    video_link_id, frames_link_id, audio_link_id = range(first_link_id, first_link_id + 3)

    shift_orders(nodes, int(r2v["order"]), 2)
    load_video = {
        "id": load_video_id,
        "type": "LoadVideo",
        "pos": [-150, 5630],
        "size": [300, 110],
        "flags": {},
        "order": int(r2v["order"]) - 2,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "VIDEO", "type": "VIDEO", "links": [video_link_id]}
        ],
        "title": "Reference Video (upload/select before running)",
        "properties": {"Node name for S&R": "LoadVideo"},
        "widgets_values": [""],
    }
    components = {
        "id": components_id,
        "type": "GetVideoComponents",
        "pos": [200, 5630],
        "size": [300, 110],
        "flags": {},
        "order": int(r2v["order"]) - 1,
        "mode": 0,
        "inputs": [
            {"name": "video", "type": "VIDEO", "link": video_link_id}
        ],
        "outputs": [
            {"name": "images", "type": "IMAGE", "links": [frames_link_id]},
            {"name": "audio", "type": "AUDIO", "links": [audio_link_id]},
            {"name": "fps", "type": "FLOAT", "links": None},
            {"name": "bit_depth", "type": "INT", "links": None},
        ],
        "properties": {"Node name for S&R": "GetVideoComponents"},
        "widgets_values": [],
    }
    nodes.extend([load_video, components])

    video_input = next(item for item in r2v["inputs"] if item["name"] == "ref_videos.ref_video_0")
    audio_input = next(item for item in r2v["inputs"] if item["name"] == "ref_video_audios.ref_video_audio_0")
    video_input["link"] = frames_link_id
    audio_input["link"] = audio_link_id
    video_slot = r2v["inputs"].index(video_input)
    audio_slot = r2v["inputs"].index(audio_input)
    links.extend(
        [
            [video_link_id, load_video_id, 0, components_id, 0, "VIDEO"],
            [frames_link_id, components_id, 0, r2v["id"], video_slot, "IMAGE"],
            [audio_link_id, components_id, 1, r2v["id"], audio_slot, "AUDIO"],
        ]
    )

    prompt = next(node for node in nodes if node["type"] == "PrimitiveStringMultiline")
    prompt["widgets_values"][0] = (
        "Use <Picture 1> for the character identity, face, and clothing. "
        "Use <Picture 2> for the environment, lighting, and visual style. "
        "Use <Video 1> for body motion, action timing, and camera movement. "
        "If <Video 1> contains audio, use its rhythm and voice characteristics as reference.\n\n"
        "Generate a new coherent cinematic shot that preserves the referenced identity and style "
        "while following the referenced motion and camera language. Keep anatomy stable, maintain "
        "temporal consistency, and generate synchronized stereo dialogue, ambience, and sound effects."
    )
    scheduler = next(node for node in nodes if node["type"] == "BasicScheduler")
    scheduler["widgets_values"][0] = "normal"

    note = next(
        node
        for node in nodes
        if node["type"] == "MarkdownNote"
        and "About this workflow" in node.get("widgets_values", [""])[0]
    )
    note["widgets_values"][0] += (
        "\n\n## Mixed video-reference input\n"
        "Upload or select a video in `Reference Video`, then `GetVideoComponents` sends its frames "
        "and paired soundtrack to `<Video 1>`. The two image loaders remain `<Picture 1>` and "
        "`<Picture 2>`. Replace, disconnect, or add optional inputs as needed."
    )

    workflow["last_node_id"] = max(int(workflow.get("last_node_id", 0)), components_id)
    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), audio_link_id)
    return workflow


def graph_candidates(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [workflow]
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        candidates.append(subgraph)
    return candidates


def link_id(link: Any) -> int:
    return int(link["id"] if isinstance(link, dict) else link[0])


def link_origin(link: Any) -> int:
    return int(link["origin_id"] if isinstance(link, dict) else link[1])


def link_target(link: Any) -> int:
    return int(link["target_id"] if isinstance(link, dict) else link[3])


def set_link_origin(link: Any, node_id: int) -> None:
    if isinstance(link, dict):
        link["origin_id"] = node_id
    else:
        link[1] = node_id


def add_easycache(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    fast = copy.deepcopy(workflow)
    fast["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{fast["id"]}:easycache'))
    graph = next(
        candidate
        for candidate in graph_candidates(fast)
        if any(node["type"] == "UNETLoader" for node in candidate.get("nodes", []))
        and any(node["type"] == "BasicGuider" for node in candidate.get("nodes", []))
    )
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    consumers = {
        int(node["id"])
        for node in nodes
        if node["type"] in {"BasicScheduler", "BasicGuider"}
    }
    patched_links = [
        link
        for link in links
        if link_origin(link) == int(unet["id"]) and link_target(link) in consumers
    ]
    if len(patched_links) != 2:
        raise RuntimeError(f"Expected UNET to feed scheduler and guider, got {len(patched_links)} links")

    cache_id = next_numeric_id(nodes)
    new_link_id = max(link_id(link) for link in links) + 1
    cache_order = int(unet["order"]) + 1
    shift_orders(nodes, cache_order, 1)
    for link in patched_links:
        set_link_origin(link, cache_id)

    old_link_ids = {link_id(link) for link in patched_links}
    unet_output = unet["outputs"][0]
    unet_output["links"] = [
        current for current in (unet_output.get("links") or []) if int(current) not in old_link_ids
    ] + [new_link_id]
    cache = {
        "id": cache_id,
        "type": "EasyCache",
        "pos": [float(unet["pos"][0]) + 700, float(unet["pos"][1])],
        "size": [360, 130],
        "flags": {},
        "order": cache_order,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": new_link_id}],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": sorted(old_link_ids)}
        ],
        "title": "EasyCache Fast (experimental)",
        "properties": {"Node name for S&R": "EasyCache"},
        "widgets_values": EASYCACHE_VALUES,
    }
    nodes.append(cache)
    if isinstance(links[0], dict):
        links.append(
            {
                "id": new_link_id,
                "origin_id": int(unet["id"]),
                "origin_slot": 0,
                "target_id": cache_id,
                "target_slot": 0,
                "type": "MODEL",
            }
        )
        state = graph.setdefault("state", {})
        state["lastNodeId"] = max(int(state.get("lastNodeId", 0)), cache_id)
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), new_link_id)
        graph["name"] = f'{graph.get("name", label)} - EasyCache Fast'
    else:
        links.append([new_link_id, int(unet["id"]), 0, cache_id, 0, "MODEL"])
        fast["last_node_id"] = max(int(fast.get("last_node_id", 0)), cache_id)
        fast["last_link_id"] = max(int(fast.get("last_link_id", 0)), new_link_id)

    note = next(
        (
            node
            for node in graph_candidates(fast)[0].get("nodes", [])
            if node["type"] == "MarkdownNote"
            and "About this workflow" in node.get("widgets_values", [""])[0]
        ),
        None,
    )
    if note:
        note["widgets_values"][0] += (
            "\n\n## EasyCache Fast variant\n"
            "Native EasyCache is enabled with `reuse_threshold=0.20`, `start_percent=0.15`, "
            "`end_percent=0.95`, and verbose logging. It skips approximate diffusion steps and "
            "may reduce motion, identity, or audio fidelity. Use the Quality workflow for finals."
        )
    return fast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2v-source", type=Path, required=True)
    parser.add_argument("--r2v-upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    i2v = read_json(args.i2v_source)
    r2v = prepare_r2v(read_json(args.r2v_upstream))
    write_json(args.output_dir / "minimax_h3_r2v.json", r2v)
    write_json(args.output_dir / "minimax_h3_i2v_easycache.json", add_easycache(i2v, "I2V"))
    write_json(args.output_dir / "minimax_h3_r2v_easycache.json", add_easycache(r2v, "R2V"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
