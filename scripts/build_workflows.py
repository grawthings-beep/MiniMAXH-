#!/usr/bin/env python3
"""Build mixed-reference, EasyCache, LoRA, and 2x upscale workflow variants."""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path
from typing import Any


EASYCACHE_VALUES = [0.2, 0.15, 0.95, True]
UPSCALER_MODEL = "RealESRGAN_x2plus.pth"
UPSCALER_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)
LORA_MODEL = "hmmotion_minimax-h3_epoch12.safetensors"
LORA_URL = (
    "https://huggingface.co/uwgm/nikke-civitai-backup/resolve/"
    f"main/{LORA_MODEL}"
)
LORA_STRENGTH = 1.0


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


def set_link_target(link: Any, node_id: int, slot: int) -> None:
    if isinstance(link, dict):
        link["target_id"] = node_id
        link["target_slot"] = slot
    else:
        link[3] = node_id
        link[4] = slot


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


def add_lora(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    """Apply the HMMotion diffusion LoRA before scheduling and guidance."""
    lora_workflow = copy.deepcopy(workflow)
    lora_workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f'{lora_workflow["id"]}:hmmotion-lora')
    )
    graph = next(
        candidate
        for candidate in graph_candidates(lora_workflow)
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
        raise RuntimeError(
            f"Expected UNET to feed scheduler and guider, got {len(patched_links)} links"
        )

    lora_id = next_numeric_id(nodes)
    new_link_id = max(link_id(link) for link in links) + 1
    lora_order = int(unet["order"]) + 1
    shift_orders(nodes, lora_order, 1)
    for link in patched_links:
        set_link_origin(link, lora_id)

    old_link_ids = {link_id(link) for link in patched_links}
    unet_output = unet["outputs"][0]
    unet_output["links"] = [
        current
        for current in (unet_output.get("links") or [])
        if int(current) not in old_link_ids
    ] + [new_link_id]
    lora = {
        "id": lora_id,
        "type": "LoraLoaderModelOnly",
        "pos": [float(unet["pos"][0]) + 690, float(unet["pos"][1])],
        "size": [430, 110],
        "flags": {},
        "order": lora_order,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "model",
                "name": "model",
                "type": "MODEL",
                "link": new_link_id,
            }
        ],
        "outputs": [
            {
                "localized_name": "MODEL",
                "name": "MODEL",
                "type": "MODEL",
                "links": sorted(old_link_ids),
            }
        ],
        "title": "HMMotion MiniMax H3 LoRA (model only)",
        "properties": {
            "Node name for S&R": "LoraLoaderModelOnly",
            "models": [
                {
                    "name": LORA_MODEL,
                    "url": LORA_URL,
                    "directory": "loras",
                }
            ],
        },
        "widgets_values": [LORA_MODEL, LORA_STRENGTH],
    }
    nodes.append(lora)
    if isinstance(links[0], dict):
        links.append(
            {
                "id": new_link_id,
                "origin_id": int(unet["id"]),
                "origin_slot": 0,
                "target_id": lora_id,
                "target_slot": 0,
                "type": "MODEL",
            }
        )
        state = graph.setdefault("state", {})
        state["lastNodeId"] = max(int(state.get("lastNodeId", 0)), lora_id)
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), new_link_id)
        graph["name"] = f'{graph.get("name", label)} - HMMotion LoRA'
    else:
        links.append([new_link_id, int(unet["id"]), 0, lora_id, 0, "MODEL"])
        lora_workflow["last_node_id"] = max(
            int(lora_workflow.get("last_node_id", 0)), lora_id
        )
        lora_workflow["last_link_id"] = max(
            int(lora_workflow.get("last_link_id", 0)), new_link_id
        )

    note = next(
        (
            node
            for node in graph_candidates(lora_workflow)[0].get("nodes", [])
            if node["type"] == "MarkdownNote"
            and "About this workflow" in node.get("widgets_values", [""])[0]
        ),
        None,
    )
    if note:
        note["widgets_values"][0] += (
            "\n\n## HMMotion LoRA\n"
            f"`{LORA_MODEL}` is applied to the diffusion model with the built-in "
            f"`LoraLoaderModelOnly` node at strength `{LORA_STRENGTH:.2f}`. The base "
            "INT8 ConvRot model feeds the LoRA loader, and the patched MODEL feeds both "
            "`BasicScheduler` and `BasicGuider`. Change the strength on the LoRA node for "
            "A/B tests; `0.0` disables its effect without rewiring the graph. The private "
            "Hugging Face asset is downloaded at Pod startup using `HF_TOKEN`."
        )
    return lora_workflow


def add_upscale(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    """Insert a conservative 2x frame upscaler before the final video mux."""
    upscaled = copy.deepcopy(workflow)
    upscaled["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{upscaled["id"]}:upscale-2x'))
    graph = next(
        candidate
        for candidate in graph_candidates(upscaled)
        if any(node["type"] == "VAEDecode" for node in candidate.get("nodes", []))
        and any(node["type"] == "CreateVideo" for node in candidate.get("nodes", []))
    )
    nodes = graph["nodes"]
    links = graph["links"]
    vae_decode = next(node for node in nodes if node["type"] == "VAEDecode")
    create_video = next(node for node in nodes if node["type"] == "CreateVideo")
    image_link = next(
        link
        for link in links
        if link_origin(link) == int(vae_decode["id"])
        and link_target(link) == int(create_video["id"])
        and str(link["type"] if isinstance(link, dict) else link[5]) == "IMAGE"
    )

    loader_id = next_numeric_id(nodes)
    upscaler_id = loader_id + 1
    first_link_id = max(link_id(link) for link in links) + 1
    model_link_id = first_link_id
    output_link_id = first_link_id + 1

    original_x = float(create_video["pos"][0])
    original_y = float(create_video["pos"][1])
    for node in nodes:
        if float(node.get("pos", [0])[0]) >= original_x:
            node["pos"][0] = float(node["pos"][0]) + 800

    create_order = int(create_video["order"])
    shift_orders(nodes, create_order, 2)
    set_link_target(image_link, upscaler_id, 1)
    next(item for item in create_video["inputs"] if item["name"] == "images")["link"] = output_link_id

    loader = {
        "id": loader_id,
        "type": "UpscaleModelLoader",
        "pos": [original_x, original_y - 210],
        "size": [300, 80],
        "flags": {},
        "order": create_order,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {
                "name": "UPSCALE_MODEL",
                "type": "UPSCALE_MODEL",
                "links": [model_link_id],
            }
        ],
        "title": "Real-ESRGAN 2x (all decoded frames)",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.10.0",
            "Node name for S&R": "UpscaleModelLoader",
            "models": [
                {
                    "name": UPSCALER_MODEL,
                    "url": UPSCALER_URL,
                    "directory": "upscale_models",
                }
            ],
        },
        "widgets_values": [UPSCALER_MODEL],
    }
    upscaler = {
        "id": upscaler_id,
        "type": "ImageUpscaleWithModel",
        "pos": [original_x + 380, original_y],
        "size": [340, 80],
        "flags": {},
        "order": create_order + 1,
        "mode": 0,
        "inputs": [
            {
                "name": "upscale_model",
                "type": "UPSCALE_MODEL",
                "link": model_link_id,
            },
            {
                "name": "image",
                "type": "IMAGE",
                "link": link_id(image_link),
            },
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [output_link_id]}
        ],
        "title": "Upscale decoded video frames 2x",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.10.0",
            "Node name for S&R": "ImageUpscaleWithModel",
        },
        "widgets_values": [],
    }
    nodes.extend([loader, upscaler])

    if isinstance(links[0], dict):
        links.extend(
            [
                {
                    "id": model_link_id,
                    "origin_id": loader_id,
                    "origin_slot": 0,
                    "target_id": upscaler_id,
                    "target_slot": 0,
                    "type": "UPSCALE_MODEL",
                },
                {
                    "id": output_link_id,
                    "origin_id": upscaler_id,
                    "origin_slot": 0,
                    "target_id": int(create_video["id"]),
                    "target_slot": 0,
                    "type": "IMAGE",
                },
            ]
        )
        state = graph.setdefault("state", {})
        state["lastNodeId"] = max(int(state.get("lastNodeId", 0)), upscaler_id)
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), output_link_id)
        graph["name"] = f'{graph.get("name", label)} - Real-ESRGAN 2x'
    else:
        links.extend(
            [
                [model_link_id, loader_id, 0, upscaler_id, 0, "UPSCALE_MODEL"],
                [output_link_id, upscaler_id, 0, int(create_video["id"]), 0, "IMAGE"],
            ]
        )
        upscaled["last_node_id"] = max(
            int(upscaled.get("last_node_id", 0)), upscaler_id
        )
        upscaled["last_link_id"] = max(
            int(upscaled.get("last_link_id", 0)), output_link_id
        )

    note = next(
        (
            node
            for node in graph_candidates(upscaled)[0].get("nodes", [])
            if node["type"] == "MarkdownNote"
            and "About this workflow" in node.get("widgets_values", [""])[0]
        ),
        None,
    )
    if note:
        note["widgets_values"][0] += (
            "\n\n## Real-ESRGAN 2x output\n"
            "Every decoded frame is tiled through the official RealESRGAN x2 model before "
            "`CreateVideo`; generated stereo audio and 24fps timing remain unchanged. The "
            "short edge doubles from H3's native 768px to 1536px. This is a conservative "
            "frame upscaler, so it improves sharpness and detail but does not repair motion "
            "or anatomy errors already present in the generated frames."
        )
    return upscaled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2v-source", type=Path, required=True)
    parser.add_argument("--r2v-upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    i2v = read_json(args.i2v_source)
    r2v = prepare_r2v(read_json(args.r2v_upstream))
    i2v_fast = add_easycache(i2v, "I2V")
    r2v_fast = add_easycache(r2v, "R2V")
    write_json(args.output_dir / "minimax_h3_r2v.json", r2v)
    write_json(args.output_dir / "minimax_h3_i2v_easycache.json", i2v_fast)
    write_json(args.output_dir / "minimax_h3_r2v_easycache.json", r2v_fast)
    write_json(args.output_dir / "minimax_h3_i2v_upscale.json", add_upscale(i2v, "I2V"))
    write_json(
        args.output_dir / "minimax_h3_i2v_hmmotion_lora_upscale.json",
        add_upscale(add_lora(i2v, "I2V"), "I2V HMMotion LoRA"),
    )
    write_json(
        args.output_dir / "minimax_h3_i2v_easycache_upscale.json",
        add_upscale(i2v_fast, "I2V EasyCache Fast"),
    )
    write_json(args.output_dir / "minimax_h3_r2v_upscale.json", add_upscale(r2v, "R2V"))
    write_json(
        args.output_dir / "minimax_h3_r2v_easycache_upscale.json",
        add_upscale(r2v_fast, "R2V EasyCache Fast"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
