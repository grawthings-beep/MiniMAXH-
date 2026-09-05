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
HMNSFW_V2_MODEL = "HMNSFW_AIO_V2.safetensors"
HMNSFW_V2_URL = (
    "https://civitai.red/api/download/models/3206518?fileId=3088013"
)
HMNSFW_V2_STRENGTH = 0.5
AUTO_MOSAIC_MODEL = "ntd11_anime_nsfw_segm_v5.pt"
AUTO_MOSAIC_URL = "https://civitai.com/api/download/models/2266294"
AUTO_MOSAIC_DEFAULTS = [
    AUTO_MOSAIC_MODEL,
    True,
    "JUST",
    0.30,
    0.50,
    0,
    3,
    "pussy,penis,testicles",
]

FIRST_BLOCK_CACHE_VALUES = [
    "H3 Safe — 0.08 / max 2",
    0.08,
    0.10,
    0.95,
    2,
    False,
]
TURBO_REVISION = "2f015e66b37c585cea9dc4ae6f1850ea8788e742"
TURBO_8STEP_MODEL = (
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
TURBO_4STEP_MODEL = (
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
)
TURBO_8STEP_URL = (
    "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/"
    f"{TURBO_REVISION}/{TURBO_8STEP_MODEL}"
)
TURBO_4STEP_URL = (
    "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/"
    f"{TURBO_REVISION}/{TURBO_4STEP_MODEL}"
)
TURBO_PROFILE_8STEP = "8-step v1.0 768p (recommended)"
TURBO_PROFILE_4STEP = "4-step v1.2 768p (fastest)"


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


def _append_model_node(
    graph: dict[str, Any],
    *,
    source: dict[str, Any],
    consumers: list[Any],
    node: dict[str, Any],
) -> None:
    """Insert a one-input MODEL patch between ``source`` and its consumers."""
    nodes = graph["nodes"]
    links = graph["links"]
    node_id = int(node["id"])
    new_link_id = max(link_id(link) for link in links) + 1
    old_link_ids = {link_id(link) for link in consumers}
    for link in consumers:
        set_link_origin(link, node_id)
    output = source["outputs"][0]
    output["links"] = [
        current
        for current in (output.get("links") or [])
        if int(current) not in old_link_ids
    ] + [new_link_id]
    node["inputs"][0]["link"] = new_link_id
    node["outputs"][0]["links"] = sorted(old_link_ids)
    nodes.append(node)
    if isinstance(links[0], dict):
        links.append(
            {
                "id": new_link_id,
                "origin_id": int(source["id"]),
                "origin_slot": 0,
                "target_id": node_id,
                "target_slot": 0,
                "type": "MODEL",
            }
        )
        state = graph.setdefault("state", {})
        state["lastNodeId"] = max(int(state.get("lastNodeId", 0)), node_id)
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), new_link_id)
    else:
        links.append([new_link_id, int(source["id"]), 0, node_id, 0, "MODEL"])


def add_first_block_cache(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    """Insert the conservative H3 FirstBlockCache directly after UNETLoader."""
    fast = copy.deepcopy(workflow)
    fast["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{fast["id"]}:first-block-cache-safe'))
    graph = next(
        candidate
        for candidate in graph_candidates(fast)
        if any(node["type"] == "UNETLoader" for node in candidate.get("nodes", []))
    )
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    consumers = [link for link in links if link_origin(link) == int(unet["id"])]
    if not consumers:
        raise RuntimeError("UNETLoader has no MODEL consumer for FirstBlockCache")

    cache_id = next_numeric_id(nodes)
    cache_order = int(unet["order"]) + 1
    shift_orders(nodes, cache_order, 1)
    cache = {
        "id": cache_id,
        "type": "ApplyMiniMaxH3FirstBlockCache",
        "pos": [-2020, 4080],
        "size": [440, 250],
        "flags": {},
        "order": cache_order,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "title": "FirstBlockCache SAFE (20 steps; replaces EasyCache)",
        "properties": {"Node name for S&R": "ApplyMiniMaxH3FirstBlockCache"},
        "widgets_values": FIRST_BLOCK_CACHE_VALUES,
    }
    _append_model_node(graph, source=unet, consumers=consumers, node=cache)
    unet["pos"] = [-2020, 3950]
    creator = next(
        (node for node in nodes if node["type"] == "LoraLoaderModelOnly"), None
    )
    if creator:
        creator["pos"] = [-2020, 4370]
    for group in graph.get("groups", []):
        if group.get("title") == "Models":
            group["bounding"] = [-2050, 3920, 700, 1420]
    graph["name"] = f'{graph.get("name", label)} - FirstBlockCache Safe'
    fast.setdefault("extra", {})["acceleration"] = {
        "type": "FirstBlockCache",
        "mode": "H3 Safe",
        "threshold": 0.08,
        "source_commit": "725973c3bfd9de6dce249bc93dc5fe27f820df31",
        "incompatible_with": ["EasyCache", "LazyCache", "CacheDiT", "T8 Block Cache"],
    }
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
            "\n\n## 02 Fast · FirstBlockCache Safe\n"
            "20 stepsを維持し、FirstBlockCacheのH3 Safe（threshold 0.08）で後段ブロックを"
            "近似再利用します。EasyCacheとは併用しません。最終採用前は01 Qualityと同一seedで比較してください。"
        )
    return fast


def add_turbo_profiles(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    """Add one-control 4/8-step 768p Turbo selection with a safe 8-step default."""
    turbo = copy.deepcopy(workflow)
    turbo["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f'{turbo["id"]}:lightx2v-turbo-4-8step-768p')
    )
    graph = next(
        candidate
        for candidate in graph_candidates(turbo)
        if any(node["type"] == "UNETLoader" for node in candidate.get("nodes", []))
    )
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    creator = next(
        node
        for node in nodes
        if node["type"] == "LoraLoaderModelOnly"
    )
    creator["widgets_values"][1] = 0.0
    creator["title"] = (
        "Optional creator LoRA (OFF by default for 32GB stability; enable manually)"
    )
    unet_consumers = [link for link in links if link_origin(link) == int(unet["id"])]
    if {link_target(link) for link in unet_consumers} != {int(creator["id"])}:
        raise RuntimeError("Turbo LoRA must be inserted before the selectable creator LoRA")

    turbo_id = next_numeric_id(nodes)
    turbo_order = int(unet["order"]) + 1
    shift_orders(nodes, turbo_order, 1)
    turbo_profile = {
        "id": turbo_id,
        "type": "MiniMaxH3TurboProfile",
        "pos": [-2020, 3930],
        "size": [520, 130],
        "flags": {},
        "order": turbo_order,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {
                "name": "profile",
                "type": "COMBO",
                "widget": {"name": "profile"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": []},
            {"name": "steps", "type": "INT", "links": []},
        ],
        "title": "Turbo Mode — select 8-step quality or 4-step fastest",
        "properties": {
            "Node name for S&R": "MiniMaxH3TurboProfile",
            "models": [
                {
                    "name": TURBO_8STEP_MODEL,
                    "url": TURBO_8STEP_URL,
                    "directory": "loras",
                },
                {
                    "name": TURBO_4STEP_MODEL,
                    "url": TURBO_4STEP_URL,
                    "directory": "loras",
                },
            ],
        },
        "widgets_values": [TURBO_PROFILE_8STEP],
    }
    _append_model_node(
        graph, source=unet, consumers=unet_consumers, node=turbo_profile
    )

    creator_consumers = [
        link for link in links if link_origin(link) == int(creator["id"])
    ]
    expected = {
        int(node["id"])
        for node in nodes
        if node["type"] in {"BasicScheduler", "BasicGuider"}
    }
    if {link_target(link) for link in creator_consumers} != expected:
        raise RuntimeError("Creator LoRA must feed BasicScheduler and BasicGuider")
    sigma_id = next_numeric_id(nodes)
    sigma_order = int(creator["order"]) + 1
    shift_orders(nodes, sigma_order, 1)
    sigma = {
        "id": sigma_id,
        "type": "MiniMaxH3SigmaShift",
        "pos": [-2020, 4230],
        "size": [360, 130],
        "flags": {},
        "order": sigma_order,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {
                "name": "shift_video",
                "type": "FLOAT",
                "widget": {"name": "shift_video"},
                "link": None,
            },
            {
                "name": "shift_audio",
                "type": "FLOAT",
                "widget": {"name": "shift_audio"},
                "link": None,
            },
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "title": "Turbo 768p Sigma Shift (video 6 / audio 3)",
        "properties": {"Node name for S&R": "MiniMaxH3SigmaShift"},
        "widgets_values": [6.0, 3.0],
    }
    _append_model_node(graph, source=creator, consumers=creator_consumers, node=sigma)

    scheduler = next(node for node in nodes if node["type"] == "BasicScheduler")
    step_link_id = max(link_id(link) for link in links) + 1
    scheduler["inputs"].append(
        {
            "name": "steps",
            "type": "INT",
            "widget": {"name": "steps"},
            "link": step_link_id,
        }
    )
    turbo_profile["outputs"][1]["links"] = [step_link_id]
    if isinstance(links[0], dict):
        links.append(
            {
                "id": step_link_id,
                "origin_id": turbo_id,
                "origin_slot": 1,
                "target_id": int(scheduler["id"]),
                "target_slot": len(scheduler["inputs"]) - 1,
                "type": "INT",
            }
        )
        state = graph.setdefault("state", {})
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), step_link_id)
    else:
        links.append(
            [
                step_link_id,
                turbo_id,
                1,
                int(scheduler["id"]),
                len(scheduler["inputs"]) - 1,
                "INT",
            ]
        )

    unet["pos"] = [-2020, 3800]
    creator["pos"] = [-2020, 4080]
    for group in graph.get("groups", []):
        if group.get("title") == "Models":
            group["bounding"] = [-2050, 3770, 700, 1570]
    sampler = next(node for node in nodes if node["type"] == "KSamplerSelect")
    sampler["widgets_values"] = ["euler"]
    scheduler["widgets_values"] = ["simple", 8, 1]
    graph["name"] = f'{graph.get("name", label)} - LightX2V Turbo 4/8-step 768p'
    turbo.setdefault("extra", {})["acceleration"] = {
        "type": "LightX2V Turbo profile selector",
        "profiles": {
            TURBO_PROFILE_8STEP: {"model": TURBO_8STEP_MODEL, "steps": 8},
            TURBO_PROFILE_4STEP: {"model": TURBO_4STEP_MODEL, "steps": 4},
        },
        "default_profile": TURBO_PROFILE_8STEP,
        "sampler": "euler",
        "scheduler": "simple",
        "shift_video": 6,
        "shift_audio": 3,
        "source_revision": TURBO_REVISION,
        "creator_lora_default_strength": 0.0,
        "memory_note": (
            "Only the selected Turbo state dict is retained; creator LoRA remains "
            "disabled by default to avoid dual-LoRA patch pressure"
        ),
    }
    note = next(
        (
            node
            for node in graph_candidates(turbo)[0].get("nodes", [])
            if node["type"] == "MarkdownNote"
            and "About this workflow" in node.get("widgets_values", [""])[0]
        ),
        None,
    )
    if note:
        note["widgets_values"][0] = note["widgets_values"][0].replace(
            "`HMNSFW_AIO_V2.safetensors` is applied to the diffusion model with the built-in "
            "`LoraLoaderModelOnly` node at strength `0.50`.",
            "`HMNSFW_AIO_V2.safetensors` remains selectable, but this Turbo preset disables "
            "the creator LoRA at strength `0.00` by default.",
        )
        note["widgets_values"][0] += (
            "\n\n## 03 Turbo · LightX2V 4/8-step 768p\n"
            "`Turbo Mode`だけで8-step v1.0（推奨）または4-step v1.2（最速）を選択できます。"
            "選択に連動して正しいLoRAとstepsが切り替わり、両方ともSigmaShift 6/3、"
            "Euler/simpleを使用します。"
            "32GBで208 patchesを避けるためcreator LoRAは初期値0.0です。必要な場合だけ手動で有効化し、"
            "二回目以降に不安定になる場合は0.0へ戻してください。音声や速い動きが崩れる場合は"
            "8-step、02 Fast、または01 Qualityへ戻してください。"
        )
    return turbo


def add_lora(
    workflow: dict[str, Any],
    label: str,
    *,
    model: str = LORA_MODEL,
    url: str = LORA_URL,
    strength: float = LORA_STRENGTH,
    variant_id: str = "hmmotion-lora",
    display_name: str = "HMMotion LoRA",
    node_title: str = "HMMotion MiniMax H3 LoRA (model only)",
    source_note: str = (
        "The private Hugging Face asset is downloaded at Pod startup using `HF_TOKEN`."
    ),
) -> dict[str, Any]:
    """Apply a selectable diffusion LoRA before scheduling and guidance."""
    lora_workflow = copy.deepcopy(workflow)
    lora_workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f'{lora_workflow["id"]}:{variant_id}')
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
        "title": node_title,
        "properties": {
            "Node name for S&R": "LoraLoaderModelOnly",
            "models": [
                {
                    "name": model,
                    "url": url,
                    "directory": "loras",
                }
            ],
        },
        "widgets_values": [model, strength],
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
        graph["name"] = f'{graph.get("name", label)} - {display_name}'
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
            f"\n\n## {display_name}\n"
            f"`{model}` is applied to the diffusion model with the built-in "
            f"`LoraLoaderModelOnly` node at strength `{strength:.2f}`. The base "
            "INT8 ConvRot model feeds the LoRA loader, and the patched MODEL feeds both "
            "`BasicScheduler` and `BasicGuider`. Change the strength on the LoRA node for "
            "A/B tests; `0.0` disables its effect without rewiring the graph. "
            f"{source_note}"
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


def add_auto_mosaic(workflow: dict[str, Any], label: str) -> dict[str, Any]:
    """Insert CPU contour mosaic once, after the final IMAGE processor."""
    patched = copy.deepcopy(workflow)
    patched["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{patched["id"]}:auto-mosaic'))
    graph = next(
        candidate
        for candidate in graph_candidates(patched)
        if any(node["type"] == "CreateVideo" for node in candidate.get("nodes", []))
    )
    nodes = graph["nodes"]
    links = graph["links"]
    create_video = next(node for node in nodes if node["type"] == "CreateVideo")
    image_input = next(item for item in create_video["inputs"] if item["name"] == "images")
    original_link_id = int(image_input["link"])
    image_link = next(link for link in links if link_id(link) == original_link_id)
    if link_target(image_link) != int(create_video["id"]):
        raise RuntimeError("CreateVideo IMAGE link target is inconsistent")

    mosaic_id = next_numeric_id(nodes)
    new_link_id = max(link_id(link) for link in links) + 1
    original_x = float(create_video["pos"][0])
    original_y = float(create_video["pos"][1])
    create_order = int(create_video["order"])

    # Move the encoder and any downstream output nodes right, leaving a clean
    # left-to-right final-frame lane for the postprocessor.
    for node in nodes:
        if float(node.get("pos", [0])[0]) >= original_x:
            node["pos"][0] = float(node["pos"][0]) + 600
    shift_orders(nodes, create_order, 1)
    set_link_target(image_link, mosaic_id, 0)
    image_input["link"] = new_link_id

    mosaic = {
        "id": mosaic_id,
        "type": "WanAutoMosaicVideo",
        "pos": [original_x, original_y - 170],
        "size": [410, 360],
        "flags": {},
        "order": create_order,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": original_link_id},
            {"name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None},
            {"name": "coverage_preset", "type": "COMBO", "widget": {"name": "coverage_preset"}, "link": None},
            {"name": "confidence", "type": "FLOAT", "widget": {"name": "confidence"}, "link": None},
            {"name": "iou_threshold", "type": "FLOAT", "widget": {"name": "iou_threshold"}, "link": None},
            {"name": "block_size", "type": "INT", "widget": {"name": "block_size"}, "link": None},
            {"name": "max_gap_frames", "type": "INT", "widget": {"name": "max_gap_frames"}, "link": None},
            {"name": "target_classes", "type": "STRING", "widget": {"name": "target_classes"}, "link": None},
        ],
        "outputs": [{"name": "mosaicked_images", "type": "IMAGE", "links": [new_link_id]}],
        "title": "AUTO MOSAIC - JUST CONTOUR (CPU, final frames)",
        "properties": {
            "Node name for S&R": "WanAutoMosaicVideo",
            "models": [
                {"name": AUTO_MOSAIC_MODEL, "url": AUTO_MOSAIC_URL, "directory": "auto_mosaic"}
            ],
        },
        "widgets_values": list(AUTO_MOSAIC_DEFAULTS),
    }
    nodes.append(mosaic)
    if isinstance(links[0], dict):
        links.append(
            {
                "id": new_link_id,
                "origin_id": mosaic_id,
                "origin_slot": 0,
                "target_id": int(create_video["id"]),
                "target_slot": 0,
                "type": "IMAGE",
            }
        )
        state = graph.setdefault("state", {})
        state["lastNodeId"] = max(int(state.get("lastNodeId", 0)), mosaic_id)
        state["lastLinkId"] = max(int(state.get("lastLinkId", 0)), new_link_id)
        graph["name"] = f'{graph.get("name", label)} - CPU Auto Mosaic'
    else:
        links.append([new_link_id, mosaic_id, 0, int(create_video["id"]), 0, "IMAGE"])
        patched["last_node_id"] = max(int(patched.get("last_node_id", 0)), mosaic_id)
        patched["last_link_id"] = max(int(patched.get("last_link_id", 0)), new_link_id)

    # Expand the existing output group instead of overlapping it with a new group.
    output_group = None
    for group in graph.get("groups", []):
        gx, gy, gw, gh = map(float, group.get("bounding", [0, 0, 0, 0]))
        if gx <= original_x <= gx + gw and gy <= original_y <= gy + gh:
            output_group = group
            break
    if output_group is None:
        output_group = next(
            (
                group for group in graph.get("groups", [])
                if "decod" in str(group.get("title", "")).lower()
                or "output" in str(group.get("title", "")).lower()
            ),
            None,
        )
    if output_group is None:
        raise RuntimeError("Final-frame graph has no output group for auto mosaic")
    gx, _gy, gw, _gh = map(float, output_group["bounding"])
    create_right = float(create_video["pos"][0]) + float(create_video["size"][0]) + 40
    output_group["bounding"][2] = max(gw, create_right - gx)
    if "Auto Mosaic" not in str(output_group.get("title", "")):
        output_group["title"] = f'{output_group.get("title", "Output")} + Auto Mosaic'

    # Earlier cache/LoRA builders preserved wiring but inherited a crowded
    # upstream position. Auto-mosaic editions are published with a clean graph.
    models_group = next(
        (group for group in graph.get("groups", []) if str(group.get("title", "")).lower() == "models"),
        None,
    )
    if models_group is not None:
        gx, gy, gw, gh = map(float, models_group["bounding"])
        for auxiliary in (
            node for node in nodes
            if node["type"] in {"EasyCache", "LoraLoaderModelOnly"}
        ):
            height = float(auxiliary["size"][1])
            auxiliary["pos"] = [gx + 30, gy - height - 50]
            new_top = float(auxiliary["pos"][1]) - 30
            old_bottom = gy + gh
            models_group["bounding"][1] = new_top
            models_group["bounding"][3] = old_bottom - new_top

    save = next((node for node in patched.get("nodes", []) if node["type"] == "SaveVideo"), None)
    if save and save.get("widgets_values"):
        save["widgets_values"][0] = str(save["widgets_values"][0]) + "_AutoMosaic"
    note = next(
        (
            node for node in patched.get("nodes", [])
            if node["type"] == "MarkdownNote"
            and "About this workflow" in node.get("widgets_values", [""])[0]
        ),
        None,
    )
    if note:
        note["widgets_values"][0] += (
            "\n\n## CPU auto mosaic variant\n"
            "`WanAutoMosaicVideo` runs exactly once on generated frames immediately before "
            "`CreateVideo` (after Real-ESRGAN when present). It uses YOLO11 instance contours "
            "on CPU with JUST / 0.30 confidence / 0.50 IoU / automatic short-edge÷50 blocks "
            "and up to 3-frame circular gap repair. Default classes are pussy, penis, and "
            "testicles; anus is deliberately excluded. Set `enabled=false` to pass the original "
            "completed frames through without loading the detector."
        )
    patched.setdefault("extra", {})["auto_mosaic"] = {
        "enabled": True,
        "stage": "final IMAGE -> WanAutoMosaicVideo -> CreateVideo",
        "cpu_only": True,
        "source_commit": "01a73bc628cc19a1df92684349285f03d4a1f39a",
    }
    return patched


def configure_ui_preset(
    workflow: dict[str, Any], *, preset: str, title: str, output_prefix: str
) -> dict[str, Any]:
    """Give a distributable preset an unmistakable canvas and output identity."""
    configured = copy.deepcopy(workflow)
    subgraph_ids = {
        str(graph["id"])
        for graph in configured.get("definitions", {}).get("subgraphs", [])
    }
    for node in configured.get("nodes", []):
        if node["type"] == "SaveVideo":
            values = list(node.get("widgets_values", []))
            if values:
                values[0] = output_prefix
                node["widgets_values"] = values
            node["title"] = f"Output · {title}"
        elif str(node["type"]) in subgraph_ids:
            node["title"] = title
    configured.setdefault("extra", {})["preset"] = preset
    configured["extra"]["ui_title"] = title
    return configured


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
    selectable_i2v = add_upscale(
        add_lora(
            i2v,
            "I2V",
            model=HMNSFW_V2_MODEL,
            url=HMNSFW_V2_URL,
            strength=HMNSFW_V2_STRENGTH,
            variant_id="hmnsfw-aio-v2-lora",
            display_name="Selectable MiniMax H3 LoRA (V2 default)",
            node_title="Selectable MiniMax H3 LoRA (V2 default; choose LoRA here)",
            source_note=(
                "Both installed creator LoRAs appear in this node's dropdown when "
                "`H3_LORA_SELECTION=all`. The Civitai V2 asset is downloaded at Pod "
                "startup using `CIVITAI_TOKEN`; its author recommends strength 0.5 or "
                "lower and trained it against BF16, so compare INT8 output carefully."
            ),
        ),
        "I2V Selectable LoRA",
    )
    variants = {
        "minimax_h3_r2v.json": r2v,
        "minimax_h3_i2v_easycache.json": i2v_fast,
        "minimax_h3_r2v_easycache.json": r2v_fast,
        "minimax_h3_i2v_upscale.json": add_upscale(i2v, "I2V"),
        "minimax_h3_i2v_hmmotion_lora_upscale.json": add_upscale(add_lora(i2v, "I2V"), "I2V HMMotion LoRA"),
        "minimax_h3_i2v_selectable_lora_upscale.json": selectable_i2v,
        "minimax_h3_i2v_easycache_upscale.json": add_upscale(i2v_fast, "I2V EasyCache Fast"),
        "minimax_h3_r2v_upscale.json": add_upscale(r2v, "R2V"),
        "minimax_h3_r2v_easycache_upscale.json": add_upscale(r2v_fast, "R2V EasyCache Fast"),
    }
    for filename, variant in variants.items():
        write_json(args.output_dir / filename, variant)

    auto_sources = {"minimax_h3_i2v.json": i2v, **variants}
    for filename, variant in auto_sources.items():
        stem = filename.removesuffix(".json")
        write_json(
            args.output_dir / f"{stem}_auto_mosaic.json",
            add_auto_mosaic(variant, stem),
        )

    quality = configure_ui_preset(
        add_auto_mosaic(selectable_i2v, "quality-preset"),
        preset="01-quality",
        title="01 · Quality · 20 steps · Selectable LoRA · 2x · Mosaic toggle",
        output_prefix="video/MiniMax_H3_01_Quality_2x",
    )
    fast = configure_ui_preset(
        add_first_block_cache(quality, "I2V Fast"),
        preset="02-fast-firstblockcache",
        title="02 · Fast · FBCache Safe · Selectable LoRA · 2x · Mosaic toggle",
        output_prefix="video/MiniMax_H3_02_Fast_FBCache_2x",
    )
    turbo = configure_ui_preset(
        add_turbo_profiles(quality, "I2V Turbo"),
        preset="03-turbo-4-8step-768p",
        title="03 · Turbo · 4/8-step 768p · Selectable LoRA · 2x · Mosaic toggle",
        output_prefix="video/MiniMax_H3_03_Turbo_4_8step_768p_2x",
    )
    presets = {
        "minimax_h3_preset_01_quality.json": quality,
        "minimax_h3_preset_02_fast_fbcache.json": fast,
        "minimax_h3_preset_03_turbo.json": turbo,
    }
    for filename, preset in presets.items():
        write_json(args.output_dir / filename, preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
