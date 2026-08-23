#!/usr/bin/env python3
"""Build ordered-keyframe MiniMax H3 Director workflow variants.

The input must be the clean ``minimax_h3_director_fl2v.json`` workflow from
``AIMixer/ComfyUI_MiniMaxH3_Director`` commit
``a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7``.  This builder deliberately
validates that small upstream graph before modifying it so a future Director
workflow change cannot silently produce a miswired graph.

Director's ``segments`` export returns list-valued IMAGE/AUDIO outputs.  The
repository's ``MiniMaxH3StoryExport2x`` OUTPUT_NODE consumes those lists in one
execution, upscales and encodes one segment at a time, releases its 2x tensor,
then ffmpeg-concats the encoded segments.  This avoids retaining an upscaled
long-form IMAGE tensor in RAM.
"""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path
from typing import Any


DIRECTOR_COMMIT = "a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7"
DIRECTOR_WORKFLOW_ID = "minimax-h3-director-fl2v"

DEFAULT_DURATION_SEC = 6.5
DEFAULT_FRAME_RATE = 24.0
DEFAULT_FRAME_COUNT = 158  # Nearest MiniMax H3 17k+5 frame count to 6.5 s @ 24 fps.
STORYBOARD_STATE = json.dumps(
    {
        "version": 1,
        "loop": False,
        "defaults": {
            "prompt": "",
            "duration_sec": DEFAULT_DURATION_SEC,
            "seed": 0,
        },
        "images": [],
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

EASYCACHE_VALUES = [0.2, 0.15, 0.95, True]
LORA_MODEL = "HMNSFW_AIO_V2.safetensors"
LORA_URL = "https://civitai.red/api/download/models/3206518?fileId=3088013"
LORA_STRENGTH = 0.5
UPSCALER_MODEL = "RealESRGAN_x2plus.pth"
UPSCALER_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)

QUALITY_OUTPUT = "minimax_h3_story_quality_lora_2x.json"
FAST_OUTPUT = "minimax_h3_story_easycache_lora_2x.json"
QUALITY_AUTO_MOSAIC_OUTPUT = "minimax_h3_story_quality_lora_2x_auto_mosaic.json"
FAST_AUTO_MOSAIC_OUTPUT = "minimax_h3_story_easycache_lora_2x_auto_mosaic.json"
AUTO_MOSAIC_MODEL = "ntd11_anime_nsfw_segm_v5.pt"
AUTO_MOSAIC_URL = "https://civitai.com/api/download/models/2266294"
AUTO_MOSAIC_DEFAULTS = [
    AUTO_MOSAIC_MODEL,
    "JUST",
    0.30,
    0.50,
    0,
    3,
    "pussy,penis,testicles",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def node_by_id(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    try:
        return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)
    except StopIteration as exc:
        raise RuntimeError(f"Pinned Director source is missing node {node_id}") from exc


def node_by_type(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = [node for node in workflow["nodes"] if node.get("type") == node_type]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {node_type} node in pinned source, got {len(matches)}"
        )
    return matches[0]


def link_by_id(workflow: dict[str, Any], wanted: int) -> list[Any]:
    try:
        return next(link for link in workflow["links"] if int(link[0]) == wanted)
    except StopIteration as exc:
        raise RuntimeError(f"Pinned Director source is missing link {wanted}") from exc


def widget_index(node: dict[str, Any], name: str) -> int:
    widget_inputs = [item for item in node["inputs"] if item.get("widget")]
    for index, item in enumerate(widget_inputs):
        if item.get("name") == name:
            # ComfyUI serializes the seed widget's control-after-generate value
            # immediately after the seed although it has no separate input row.
            if node.get("type") == "MiniMaxH3Director" and index > 4:
                return index + 1
            return index
    raise RuntimeError(f"Node {node['id']} has no widget input named {name!r}")


def validate_clean_source(workflow: dict[str, Any]) -> None:
    """Reject anything except the pinned clean 11-node FL2V topology."""

    if workflow.get("id") != DIRECTOR_WORKFLOW_ID:
        raise RuntimeError(
            f"Expected workflow id {DIRECTOR_WORKFLOW_ID!r} from {DIRECTOR_COMMIT}"
        )
    expected_types = {
        1: "UNETLoader",
        2: "CLIPLoader",
        3: "VAELoader",
        4: "VAELoader",
        5: "MiniMaxH3Director",
        6: "CreateVideo",
        7: "SaveVideo",
        8: "PreviewAny",
        9: "PreviewAny",
        10: "PreviewAny",
        11: "MarkdownNote",
    }
    actual = {int(node["id"]): node.get("type") for node in workflow.get("nodes", [])}
    if actual != expected_types:
        raise RuntimeError(
            "Director source node set differs from the pinned clean FL2V workflow"
        )
    if not workflow.get("links") or any(
        not isinstance(link, list) or len(link) != 6 for link in workflow["links"]
    ):
        raise RuntimeError("Pinned Director source must use six-field list links")
    expected_links = {
        1: (1, 0, 5, 0, "MODEL"),
        2: (2, 0, 5, 3, "CLIP"),
        3: (3, 0, 5, 1, "VAE"),
        4: (4, 0, 5, 2, "VAE"),
        5: (5, 0, 6, 0, "IMAGE"),
        6: (5, 1, 6, 2, "AUDIO"),
        7: (5, 2, 6, 1, "FLOAT"),
        8: (5, 5, 8, 0, "*"),
        9: (6, 0, 7, 0, "VIDEO"),
        10: (5, 2, 9, 0, "*"),
        11: (5, 3, 10, 0, "*"),
    }
    actual_links = {
        int(link[0]): (int(link[1]), int(link[2]), int(link[3]), int(link[4]), link[5])
        for link in workflow["links"]
    }
    if actual_links != expected_links:
        raise RuntimeError(
            "Director source link set differs from the pinned clean FL2V workflow"
        )

    director = node_by_id(workflow, 5)
    if [item["name"] for item in director["inputs"][:4]] != [
        "model",
        "video_vae",
        "audio_vae",
        "clip",
    ]:
        raise RuntimeError("Director source core input order is not the pinned topology")
    if any(item.get("name") in {"i2v_groups", "r2v_groups"} for item in director["inputs"]):
        raise RuntimeError("Director source is not clean: external group sockets already exist")
    if len(director.get("widgets_values", [])) != 21:
        raise RuntimeError("Director source widget layout is not the pinned 21-value layout")


def configure_director(director: dict[str, Any], storyboard_link_id: int) -> None:
    """Add the external group sockets and safe per-segment export defaults."""

    director["inputs"][4:4] = [
        {
            "name": "i2v_groups",
            "type": "MMX_DIR_GROUP",
            "link": storyboard_link_id,
        },
        {"name": "r2v_groups", "type": "MMX_DIR_GROUP", "link": None},
    ]
    timeline_index = widget_index(director, "timeline_data")
    timeline = json.loads(director["widgets_values"][timeline_index])
    output = timeline.setdefault("output", {})
    output["exportMode"] = "segments"
    # Director embeds the H3 Motion Context mechanism. Carry the previous
    # segment's final latent/audio motion into the next segment, then trim the
    # duplicated head before export.
    output["continuityEnabled"] = True
    output["continuityOverlapFrames"] = 22
    output["maxExportFrames"] = 0
    timeline["totalFrames"] = DEFAULT_FRAME_COUNT
    timeline["frameRate"] = DEFAULT_FRAME_RATE
    timeline.setdefault("gen", {})["defaultFrameCount"] = DEFAULT_FRAME_COUNT
    if timeline.get("segments"):
        segment = timeline["segments"][0]
        segment["length"] = DEFAULT_FRAME_COUNT
        segment["frameCount"] = DEFAULT_FRAME_COUNT
        segment["durationSec"] = DEFAULT_DURATION_SEC
    director["widgets_values"][timeline_index] = json.dumps(
        timeline, ensure_ascii=False, separators=(",", ":")
    )
    director["widgets_values"][widget_index(director, "total_frames")] = (
        DEFAULT_FRAME_COUNT
    )
    director["widgets_values"][widget_index(director, "frame_rate")] = (
        DEFAULT_FRAME_RATE
    )
    director["widgets_values"][
        widget_index(director, "clear_vram_between_segments")
    ] = True


def make_lora(node_id: int, input_link: int, output_links: list[int]) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "LoraLoaderModelOnly",
        "pos": [-1080, -240],
        "size": [430, 110],
        "flags": {},
        "order": 4,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "model",
                "name": "model",
                "type": "MODEL",
                "link": input_link,
            }
        ],
        "outputs": [
            {
                "localized_name": "MODEL",
                "name": "MODEL",
                "type": "MODEL",
                "links": output_links,
            }
        ],
        "title": "Selectable MiniMax H3 LoRA (HMNSFW V2 default; choose here)",
        "properties": {
            "Node name for S&R": "LoraLoaderModelOnly",
            "models": [
                {"name": LORA_MODEL, "url": LORA_URL, "directory": "loras"}
            ],
        },
        "widgets_values": [LORA_MODEL, LORA_STRENGTH],
    }


def make_storyboard(node_id: int, output_link: int, order: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "MiniMaxH3OrderedStoryboard",
        "pos": [-1520, 320],
        "size": [760, 860],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {
                "name": "storyboard_data",
                "type": "STRING",
                "widget": {"name": "storyboard_data"},
                "link": None,
            }
        ],
        "outputs": [
            {"name": "groups", "type": "MMX_DIR_GROUP", "links": [output_link]},
            {"name": "storyboard_plan", "type": "STRING", "links": [15]},
            {"name": "segment_count", "type": "INT", "links": None},
            {"name": "total_seconds", "type": "FLOAT", "links": None},
        ],
        "title": "Ordered Storyboard（画像順・区間プロンプト・Loop）",
        "properties": {"Node name for S&R": "MiniMaxH3OrderedStoryboard"},
        "widgets_values": [STORYBOARD_STATE],
    }


def make_easycache(node_id: int, input_link: int, output_link: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "EasyCache",
        "pos": [-600, -240],
        "size": [360, 130],
        "flags": {},
        "order": 5,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": input_link}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [output_link]}],
        "title": "EasyCache Fast (experimental)",
        "properties": {"Node name for S&R": "EasyCache"},
        "widgets_values": EASYCACHE_VALUES,
    }


def make_upscale_loader(node_id: int, output_link: int, order: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "UpscaleModelLoader",
        "pos": [540, 320],
        "size": [320, 82],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": [output_link]}
        ],
        "title": "Real-ESRGAN 2x model",
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


def make_story_exporter(node_id: int, order: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "MiniMaxH3StoryExport2x",
        "pos": [900, 320],
        "size": [460, 300],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": 5},
            {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": 14},
            {
                "name": "fps",
                "type": "FLOAT",
                "widget": {"name": "fps"},
                "link": 7,
            },
            {
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
                "link": None,
            },
            {
                "name": "crf",
                "type": "INT",
                "widget": {"name": "crf"},
                "link": None,
            },
            {
                "name": "preset",
                "type": "COMBO",
                "widget": {"name": "preset"},
                "link": None,
            },
            {
                "name": "drop_boundary_duplicates",
                "type": "BOOLEAN",
                "widget": {"name": "drop_boundary_duplicates"},
                "link": None,
            },
            {
                "name": "drop_loop_terminal",
                "type": "BOOLEAN",
                "widget": {"name": "drop_loop_terminal"},
                "link": None,
            },
            {"name": "audio", "type": "AUDIO", "link": 6},
            {
                "name": "storyboard_plan",
                "type": "STRING",
                "link": 15,
                "shape": 7,
            },
        ],
        "outputs": [
            {"name": "video", "type": "VIDEO", "links": None},
            {"name": "saved_path", "type": "STRING", "links": None},
        ],
        "title": "2x Story Export (memory bounded, final OUTPUT_NODE)",
        "properties": {"Node name for S&R": "MiniMaxH3StoryExport2x"},
        "widgets_values": [
            DEFAULT_FRAME_RATE,
            "video/MiniMax_H3_Story_2x",
            18,
            "fast",
            True,
            True,
        ],
    }


def make_auto_mosaic(node_id: int, input_link: int, output_link: int, order: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "WanAutoMosaicVideo",
        "pos": [900, 320],
        "size": [410, 360],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": input_link},
            {"name": "model_name", "type": "COMBO", "widget": {"name": "model_name"}, "link": None},
            {"name": "coverage_preset", "type": "COMBO", "widget": {"name": "coverage_preset"}, "link": None},
            {"name": "confidence", "type": "FLOAT", "widget": {"name": "confidence"}, "link": None},
            {"name": "iou_threshold", "type": "FLOAT", "widget": {"name": "iou_threshold"}, "link": None},
            {"name": "block_size", "type": "INT", "widget": {"name": "block_size"}, "link": None},
            {"name": "max_gap_frames", "type": "INT", "widget": {"name": "max_gap_frames"}, "link": None},
            {"name": "target_classes", "type": "STRING", "widget": {"name": "target_classes"}, "link": None},
        ],
        "outputs": [{"name": "mosaicked_images", "type": "IMAGE", "links": [output_link]}],
        "title": "AUTO MOSAIC - JUST CONTOUR (CPU, generated segments)",
        "properties": {
            "Node name for S&R": "WanAutoMosaicVideo",
            "models": [
                {"name": AUTO_MOSAIC_MODEL, "url": AUTO_MOSAIC_URL, "directory": "auto_mosaic"}
            ],
        },
        "widgets_values": list(AUTO_MOSAIC_DEFAULTS),
    }


def normalize_orders(workflow: dict[str, Any], fast: bool) -> None:
    orders = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        12: 4,
        13: 6 if fast else 5,
        5: 7 if fast else 6,
        14: 8 if fast else 7,
        15: 9 if fast else 8,
        8: 10 if fast else 9,
        9: 11 if fast else 10,
        10: 12 if fast else 11,
        11: 13 if fast else 12,
    }
    if fast:
        orders[16] = 5
    for node in workflow["nodes"]:
        node["order"] = orders[int(node["id"])]


def update_note(workflow: dict[str, Any], fast: bool) -> None:
    note = node_by_type(workflow, "MarkdownNote")
    speed = (
        "\n- **Fast版**: EasyCache 0.20 / 0.15–0.95をLoRA後に適用。"
        "品質低下時はQuality版を使用。"
        if fast
        else "\n- **Quality版**: EasyCacheなし。最終出力向け。"
    )
    note["title"] = "MiniMax H3 Ordered Storyboard"
    note["widgets_values"] = [
        "## MiniMax H3 · Ordered Storyboard\n\n"
        "1. Ordered Storyboardへ2枚以上を追加し、↑↓で順番を変更\n"
        "2. 各画像から次の画像までのprompt / duration / seedを指定\n"
        "3. 通常は隣接ペア、Loop ONなら最後→最初も生成\n"
        "4. LoRAはHMNSFW V2 / 0.5が初期値。ノードのプルダウンで変更可能\n"
        "5. Motion Context ON（22f）: 前区間のlatent＋音声を次区間へ継承し、重複headをTrim\n"
        "6. 専用OUTPUTノードが各区間をReal-ESRGAN 2xして即時encodeし、音声付きで結合保存\n"
        "7. DirectorのRAM保護のため全区間の合計は90秒まで\n"
        f"{speed}\n\n"
        "### プロンプトのコツ\n"
        "境界画像で停止させず、動作を次の区間へ継続するよう明記します。"
        "例: `The subject passes through the key pose without pausing; motion and camera momentum continue.`\n\n"
        "### メモリ設計\n"
        "MiniMaxH3StoryExport2xはDirectorの全区間リストを一度だけ受け、各区間を16fずつ"
        "2x→一時MKV化→テンソル解放してからffmpeg concatします。境界の重複フレームと"
        "loop終端の先頭フレームは既定で除去します。"
    ]
    note["pos"] = [540, 1080]
    note["size"] = [620, 430]


def validate_generated(workflow: dict[str, Any], fast: bool, auto_mosaic: bool = False) -> None:
    """Static, bidirectional link validation plus requested graph invariants."""

    node_ids = {int(node["id"]) for node in workflow["nodes"]}
    if len(node_ids) != len(workflow["nodes"]):
        raise RuntimeError("Generated workflow has duplicate node ids")
    links = {int(link[0]): link for link in workflow["links"]}
    if len(links) != len(workflow["links"]):
        raise RuntimeError("Generated workflow has duplicate link ids")
    for link_id, link in links.items():
        if int(link[1]) not in node_ids or int(link[3]) not in node_ids:
            raise RuntimeError(f"Link {link_id} references a missing node")
        origin = node_by_id(workflow, int(link[1]))
        target = node_by_id(workflow, int(link[3]))
        if link_id not in (origin["outputs"][int(link[2])].get("links") or []):
            raise RuntimeError(f"Link {link_id} missing from origin node output")
        if int(target["inputs"][int(link[4])].get("link") or -1) != link_id:
            raise RuntimeError(f"Link {link_id} missing from target node input")

    mosaics = [node for node in workflow["nodes"] if node["type"] == "WanAutoMosaicVideo"]
    if len(mosaics) != (1 if auto_mosaic else 0):
        raise RuntimeError("Auto-mosaic node count does not match the workflow variant")

    expected_model_path = ["UNETLoader", "LoraLoaderModelOnly"]
    if fast:
        expected_model_path.append("EasyCache")
    expected_model_path.append("MiniMaxH3Director")
    current = node_by_type(workflow, "UNETLoader")
    actual_model_path = [current["type"]]
    while current["type"] != "MiniMaxH3Director":
        model_links = current["outputs"][0].get("links") or []
        if len(model_links) != 1:
            raise RuntimeError("Generated MODEL chain is not linear")
        current = node_by_id(workflow, int(links[int(model_links[0])][3]))
        actual_model_path.append(current["type"])
    if actual_model_path != expected_model_path:
        raise RuntimeError(f"Unexpected MODEL path: {actual_model_path!r}")

    director = node_by_type(workflow, "MiniMaxH3Director")
    i2v = next(item for item in director["inputs"] if item["name"] == "i2v_groups")
    storyboard_link = links[int(i2v["link"])]
    if node_by_id(workflow, int(storyboard_link[1]))["type"] != "MiniMaxH3OrderedStoryboard":
        raise RuntimeError("Ordered Storyboard is not wired to Director.i2v_groups")
    timeline = json.loads(
        director["widgets_values"][widget_index(director, "timeline_data")]
    )
    if timeline["output"].get("exportMode") != "segments":
        raise RuntimeError("Director exportMode must be segments")
    if timeline["output"].get("continuityEnabled") is not True:
        raise RuntimeError("Director Motion Context continuity must be enabled")
    if int(timeline["output"].get("continuityOverlapFrames", 0)) != 22:
        raise RuntimeError("Director Motion Context must use the recommended 22 frames")
    if director["widgets_values"][
        widget_index(director, "clear_vram_between_segments")
    ] is not True:
        raise RuntimeError("Director must clear VRAM between segments")

    lora = node_by_type(workflow, "LoraLoaderModelOnly")
    if lora["widgets_values"] != [LORA_MODEL, LORA_STRENGTH]:
        raise RuntimeError("Generated workflow has the wrong default LoRA")
    if fast != any(node["type"] == "EasyCache" for node in workflow["nodes"]):
        raise RuntimeError("EasyCache presence does not match requested variant")

    forbidden_tail = {"ImageUpscaleWithModel", "CreateVideo", "SaveVideo"}
    if any(node["type"] in forbidden_tail for node in workflow["nodes"]):
        raise RuntimeError("Legacy in-memory video tail remains in generated workflow")
    exporter = node_by_type(workflow, "MiniMaxH3StoryExport2x")
    expected_export_inputs = [
        "images",
        "upscale_model",
        "fps",
        "filename_prefix",
        "crf",
        "preset",
        "drop_boundary_duplicates",
        "drop_loop_terminal",
        "audio",
        "storyboard_plan",
    ]
    if [item["name"] for item in exporter["inputs"]] != expected_export_inputs:
        raise RuntimeError("Story exporter inputs do not match the registered node schema")
    image_link = 5
    if auto_mosaic:
        mosaic = mosaics[0]
        if mosaic.get("widgets_values") != AUTO_MOSAIC_DEFAULTS:
            raise RuntimeError("Story auto-mosaic defaults changed unexpectedly")
        if next(item for item in mosaic["inputs"] if item["name"] == "images")["link"] != 5:
            raise RuntimeError("Director IMAGE segments are not wired into auto mosaic")
        output_links = mosaic["outputs"][0].get("links") or []
        if len(output_links) != 1:
            raise RuntimeError("Story auto mosaic must feed the exporter exactly once")
        image_link = int(output_links[0])
        if int(links[5][3]) != int(mosaic["id"]):
            raise RuntimeError("Director IMAGE link bypasses story auto mosaic")
        if int(links[image_link][3]) != int(exporter["id"]):
            raise RuntimeError("Story auto mosaic output does not feed the exporter")
    expected_export_links = [image_link, 14, 7, None, None, None, None, None, 6, 15]
    if [item.get("link") for item in exporter["inputs"]] != expected_export_links:
        raise RuntimeError("Story exporter is not wired to all Director/storyboard outputs")
    if exporter["widgets_values"][2:] != [18, "fast", True, True]:
        raise RuntimeError("Story exporter encoding/boundary defaults changed unexpectedly")

    groups = workflow.get("groups", [])
    group_ids = [int(group["id"]) for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("Generated workflow has duplicate group ids")

    def rectangle(item: dict[str, Any]) -> tuple[float, float, float, float]:
        x, y, width, height = map(float, item["bounding"])
        return x, y, x + width, y + height

    group_rectangles = [(group, rectangle(group)) for group in groups]
    for index, (left_group, left) in enumerate(group_rectangles):
        for right_group, right in group_rectangles[index + 1 :]:
            overlaps = (
                min(left[2], right[2]) > max(left[0], right[0])
                and min(left[3], right[3]) > max(left[1], right[1])
            )
            if overlaps:
                raise RuntimeError(
                    f"Workflow groups overlap: {left_group['title']!r} and "
                    f"{right_group['title']!r}"
                )

    for node in workflow["nodes"]:
        x, y = map(float, node["pos"])
        width, height = map(float, node["size"])
        contained = [
            group["title"]
            for group, bounds in group_rectangles
            if x >= bounds[0]
            and y >= bounds[1]
            and x + width <= bounds[2]
            and y + height <= bounds[3]
        ]
        if len(contained) != 1:
            raise RuntimeError(
                f"Node {node['id']} ({node['type']}) must be inside exactly one group; "
                f"found {contained!r}"
            )


def build_variant(source: dict[str, Any], *, fast: bool) -> dict[str, Any]:
    validate_clean_source(source)
    workflow = copy.deepcopy(source)
    suffix = "story-easycache-lora-2x" if fast else "story-quality-lora-2x"
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{DIRECTOR_COMMIT}:{suffix}"))
    workflow["revision"] = 0

    unet = node_by_id(workflow, 1)
    director = node_by_id(workflow, 5)

    # The pinned example's CreateVideo/SaveVideo pair expects one in-memory
    # IMAGE batch.  Replace it with the repository OUTPUT_NODE, which owns the
    # Director list and processes one segment at a time.
    workflow["nodes"] = [
        node for node in workflow["nodes"] if int(node["id"]) not in {6, 7}
    ]
    workflow["links"] = [link for link in workflow["links"] if int(link[0]) != 9]

    # Reuse upstream link 1 as the final MODEL -> Director edge.  Add link 12
    # for UNET -> LoRA, and link 16 only in the EasyCache variant.
    model_to_director = link_by_id(workflow, 1)
    model_to_director[1] = 16 if fast else 12
    unet["outputs"][0]["links"] = [12]

    storyboard_link_id = 13
    configure_director(director, storyboard_link_id)

    # Reuse the Director example's IMAGE/AUDIO/FLOAT links, retargeting them to
    # the matching INPUT_IS_LIST exporter sockets.
    image_to_exporter = link_by_id(workflow, 5)
    image_to_exporter[3] = 15
    image_to_exporter[4] = 0
    audio_to_exporter = link_by_id(workflow, 6)
    audio_to_exporter[3] = 15
    audio_to_exporter[4] = 8
    fps_to_exporter = link_by_id(workflow, 7)
    fps_to_exporter[3] = 15
    fps_to_exporter[4] = 2

    workflow["links"].extend(
        [
            [12, 1, 0, 12, 0, "MODEL"],
            [13, 13, 0, 5, 4, "MMX_DIR_GROUP"],
            [14, 14, 0, 15, 1, "UPSCALE_MODEL"],
            [15, 13, 1, 15, 9, "STRING"],
        ]
    )

    lora_output = 16 if fast else 1
    workflow["nodes"].extend(
        [
            make_lora(12, 12, [lora_output]),
            make_storyboard(13, 13, 6 if fast else 5),
            make_upscale_loader(14, 14, 8 if fast else 7),
            make_story_exporter(15, 9 if fast else 8),
        ]
    )
    if fast:
        workflow["links"].append([16, 12, 0, 16, 0, "MODEL"])
        workflow["nodes"].append(make_easycache(16, 16, 1))

    exporter = node_by_id(workflow, 15)
    exporter["widgets_values"][1] = (
        "video/MiniMax_H3_Story_EasyCache_LoRA_2x"
        if fast
        else "video/MiniMax_H3_Story_Quality_LoRA_2x"
    )
    update_note(workflow, fast)

    # Four non-overlapping lanes: models, storyboard, generation, output.
    positions = {
        1: [-1520, -240],
        2: [-1520, -120],
        3: [-1520, 30],
        4: [-1520, 130],
        5: [-640, 320],
        8: [540, 700],
        9: [1220, 700],
        10: [1460, 700],
    }
    for node_id, position in positions.items():
        node_by_id(workflow, node_id)["pos"] = position
    director["size"] = [1060, 1000]

    normalize_orders(workflow, fast)
    workflow["last_node_id"] = 16 if fast else 15
    workflow["last_link_id"] = 16 if fast else 15
    workflow.setdefault("extra", {})["note"] = (
        "MiniMax H3 ordered storyboard · EasyCache + selectable LoRA + 2x"
        if fast
        else "MiniMax H3 ordered storyboard · Quality + selectable LoRA + 2x"
    )

    # Replace the overlapping upstream groups with clean, numbered lanes.
    for group in workflow.get("groups", []):
        if int(group.get("id", 0)) == 1:
            group["title"] = "1 · Models / LoRA / Acceleration"
            group["bounding"] = [-1560, -280, 1400, 500]
        elif int(group.get("id", 0)) == 2:
            group["title"] = "3 · MiniMax H3 Director · Motion Context 22f"
            group["bounding"] = [-680, 280, 1140, 1100]
        elif int(group.get("id", 0)) == 3:
            group["title"] = "4 · 2x Output"
            group["bounding"] = [500, 280, 1240, 720]
    workflow.setdefault("groups", []).extend(
        [
            {
                "id": 4,
                "title": "2 · Storyboard Editor",
                "bounding": [-1560, 280, 820, 980],
                "color": "#6d5cae",
                "flags": {},
            },
            {
                "id": 5,
                "title": "Guide / Prompt Tips",
                "bounding": [500, 1040, 700, 510],
                "color": "#3f789e",
                "flags": {},
            },
        ]
    )

    validate_generated(workflow, fast, auto_mosaic=False)
    return workflow


def add_auto_mosaic(workflow: dict[str, Any], *, fast: bool) -> dict[str, Any]:
    """Insert CPU contour processing between Director segments and final export."""
    patched = copy.deepcopy(workflow)
    patched["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{patched["id"]}:auto-mosaic'))
    director = node_by_type(patched, "MiniMaxH3Director")
    exporter = node_by_type(patched, "MiniMaxH3StoryExport2x")
    image_link = link_by_id(patched, 5)
    if int(image_link[1]) != int(director["id"]) or int(image_link[3]) != int(exporter["id"]):
        raise RuntimeError("Pinned Director-to-exporter IMAGE link changed")

    mosaic_id = max(int(node["id"]) for node in patched["nodes"]) + 1
    output_link = max(int(link[0]) for link in patched["links"]) + 1
    export_order = int(exporter["order"])
    for node in patched["nodes"]:
        if int(node.get("order", -1)) >= export_order:
            node["order"] = int(node["order"]) + 1

    image_link[3] = mosaic_id
    image_link[4] = 0
    exporter["inputs"][0]["link"] = output_link
    exporter["pos"] = [1360, 320]
    exporter["widgets_values"][1] = str(exporter["widgets_values"][1]) + "_AutoMosaic"
    patched["nodes"].append(make_auto_mosaic(mosaic_id, 5, output_link, export_order))
    patched["links"].append([output_link, mosaic_id, 0, int(exporter["id"]), 0, "IMAGE"])

    # Keep the output lane legible without node or group overlap.
    node_by_id(patched, 8)["pos"] = [540, 760]
    node_by_id(patched, 9)["pos"] = [1220, 760]
    node_by_id(patched, 10)["pos"] = [1460, 760]
    for group in patched.get("groups", []):
        group_id = int(group.get("id", 0))
        if group_id == 1:
            group["bounding"] = [-1560, -280, 1400, 500]
        elif group_id == 2:
            group["bounding"] = [-680, 280, 1140, 1100]
        elif group_id == 3:
            group["title"] = "4 · Auto Mosaic (CPU) → 2x Output"
            group["bounding"] = [500, 280, 1420, 720]

    note = node_by_type(patched, "MarkdownNote")
    note["widgets_values"][0] += (
        "\n\n## CPU auto mosaic\n"
        "Generated Director IMAGE segments pass through `WanAutoMosaicVideo` exactly once "
        "before the memory-bounded 2x/MP4 exporter. Detection and contour pixelation are "
        "CPU-only. JUST, confidence 0.30, IoU 0.50, automatic short-edge÷50 blocks, and "
        "3-frame circular gap repair are the defaults; anus is deliberately excluded."
    )
    patched["last_node_id"] = mosaic_id
    patched["last_link_id"] = output_link
    patched.setdefault("extra", {})["auto_mosaic"] = {
        "enabled": True,
        "stage": "Director final IMAGE segments -> WanAutoMosaicVideo -> 2x/MP4 exporter",
        "cpu_only": True,
        "source_commit": "01a73bc628cc19a1df92684349285f03d4a1f39a",
    }
    validate_generated(patched, fast, auto_mosaic=True)
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--director-source",
        type=Path,
        required=True,
        help=(
            "Clean minimax_h3_director_fl2v.json copied from AIMixer Director "
            f"commit {DIRECTOR_COMMIT}"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = read_json(args.director_source)
    quality = build_variant(source, fast=False)
    fast = build_variant(source, fast=True)
    write_json(args.output_dir / QUALITY_OUTPUT, quality)
    write_json(args.output_dir / FAST_OUTPUT, fast)
    write_json(
        args.output_dir / QUALITY_AUTO_MOSAIC_OUTPUT,
        add_auto_mosaic(quality, fast=False),
    )
    write_json(
        args.output_dir / FAST_AUTO_MOSAIC_OUTPUT,
        add_auto_mosaic(fast, fast=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
