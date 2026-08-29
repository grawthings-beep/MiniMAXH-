#!/usr/bin/env python3
"""Verify MiniMax H3 workflows, model sets, and native ComfyUI nodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRONTEND_BUILTINS = {"MarkdownNote"}
HMMOTION_LORA = "hmmotion_minimax-h3_epoch12.safetensors"
HMNSFW_V2_LORA = "HMNSFW_AIO_V2.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
AUTO_MOSAIC_MODEL = "auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"
AUTO_MOSAIC_DEFAULTS = [
    "ntd11_anime_nsfw_segm_v5.pt",
    True,
    "JUST",
    0.30,
    0.50,
    0,
    3,
    "pussy,penis,testicles",
]


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
    comfyui_root: Path,
    node_types: set[str],
    subgraph_ids: set[str],
    custom_node_roots: list[Path] | None = None,
) -> None:
    source_files = [comfyui_root / "nodes.py", *sorted((comfyui_root / "comfy_extras").glob("*.py"))]
    for root in custom_node_roots or []:
        if not root.is_dir():
            raise RuntimeError(f"Custom-node source root does not exist: {root}")
        source_files.extend(sorted(root.rglob("*.py")))
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


def link_type(link: object) -> str:
    return str(link["type"] if isinstance(link, dict) else link[5])


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


def verify_upscale_wiring(
    workflow: dict[str, object], *, expect_auto_mosaic: bool = False
) -> None:
    graph = next(
        (
            candidate
            for candidate in graph_candidates(workflow)
            if any(node["type"] == "UpscaleModelLoader" for node in candidate.get("nodes", []))
            and any(node["type"] == "ImageUpscaleWithModel" for node in candidate.get("nodes", []))
        ),
        None,
    )
    if graph is None:
        raise RuntimeError("The 2x upscale nodes are missing")
    nodes = graph["nodes"]
    links = graph["links"]
    vae_decode = next(node for node in nodes if node["type"] == "VAEDecode")
    loader = next(node for node in nodes if node["type"] == "UpscaleModelLoader")
    upscaler = next(node for node in nodes if node["type"] == "ImageUpscaleWithModel")
    create_video = next(node for node in nodes if node["type"] == "CreateVideo")
    expected_target = next(
        node for node in nodes
        if node["type"] == ("WanAutoMosaicVideo" if expect_auto_mosaic else "CreateVideo")
    )
    expected = {
        (int(loader["id"]), int(upscaler["id"]), "UPSCALE_MODEL"),
        (int(vae_decode["id"]), int(upscaler["id"]), "IMAGE"),
        (int(upscaler["id"]), int(expected_target["id"]), "IMAGE"),
    }
    actual = {(link_origin(link), link_target(link), link_type(link)) for link in links}
    if not expected <= actual:
        raise RuntimeError(f"2x upscale wiring is incomplete: {sorted(expected - actual)}")
    if (int(vae_decode["id"]), int(create_video["id"]), "IMAGE") in actual:
        raise RuntimeError("VAEDecode still bypasses the 2x upscaler")
    if loader.get("widgets_values") != ["RealESRGAN_x2plus.pth"]:
        raise RuntimeError("The 2x upscaler model selection has changed")
    model_entries = loader.get("properties", {}).get("models", [])
    if len(model_entries) != 1 or model_entries[0].get("directory") != "upscale_models":
        raise RuntimeError("The 2x upscaler model metadata is incomplete")


def verify_lora_wiring(
    workflow: dict[str, object], expected_model: str, expected_strength: float
) -> None:
    graph = next(
        (
            candidate
            for candidate in graph_candidates(workflow)
            if any(
                node["type"] == "LoraLoaderModelOnly"
                for node in candidate.get("nodes", [])
            )
        ),
        None,
    )
    if graph is None:
        raise RuntimeError("The LoRA loader is missing")
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    lora = next(
        node
        for node in nodes
        if node["type"] == "LoraLoaderModelOnly"
        and node.get("widgets_values", [None])[0] == expected_model
    )
    scheduler = next(node for node in nodes if node["type"] == "BasicScheduler")
    guider = next(node for node in nodes if node["type"] == "BasicGuider")
    model_edges = {
        (link_origin(link), link_target(link))
        for link in links
        if link_type(link) == "MODEL"
    }

    def reachable(start: int, target: int) -> bool:
        pending = [start]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(dst for src, dst in model_edges if src == current)
        return False

    if not reachable(int(unet["id"]), int(lora["id"])):
        raise RuntimeError("UNETLoader does not reach the selected creator LoRA")
    if not reachable(int(lora["id"]), int(scheduler["id"])) or not reachable(
        int(lora["id"]), int(guider["id"])
    ):
        raise RuntimeError("Creator LoRA does not reach both scheduler and guider")
    if lora.get("widgets_values") != [expected_model, expected_strength]:
        raise RuntimeError(
            f"LoRA defaults have changed; expected {expected_model} at {expected_strength}"
        )
    model_entries = lora.get("properties", {}).get("models", [])
    if len(model_entries) != 1 or model_entries[0].get("directory") != "loras":
        raise RuntimeError("The LoRA model metadata is incomplete")


def verify_first_block_cache(workflow: dict[str, object]) -> None:
    graph = graph_with_types(
        workflow, {"UNETLoader", "ApplyMiniMaxH3FirstBlockCache", "LoraLoaderModelOnly"}
    )
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    cache = next(
        node for node in nodes if node["type"] == "ApplyMiniMaxH3FirstBlockCache"
    )
    creator = next(node for node in nodes if node["type"] == "LoraLoaderModelOnly")
    actual = {(link_origin(link), link_target(link), link_type(link)) for link in links}
    expected = {
        (int(unet["id"]), int(cache["id"]), "MODEL"),
        (int(cache["id"]), int(creator["id"]), "MODEL"),
    }
    if not expected <= actual:
        raise RuntimeError(f"FirstBlockCache wiring is incomplete: {sorted(expected - actual)}")
    if cache.get("widgets_values") != [
        "H3 Safe — 0.08 / max 2", 0.08, 0.10, 0.95, 2, False
    ]:
        raise RuntimeError("FirstBlockCache must use the conservative H3 Safe preset")
    if any(node["type"] == "EasyCache" for node in nodes):
        raise RuntimeError("FirstBlockCache and EasyCache must never be combined")


def verify_turbo_8step(workflow: dict[str, object]) -> None:
    graph = graph_with_types(
        workflow,
        {"UNETLoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "BasicScheduler", "KSamplerSelect"},
    )
    nodes = graph["nodes"]
    links = graph["links"]
    unet = next(node for node in nodes if node["type"] == "UNETLoader")
    loras = [node for node in nodes if node["type"] == "LoraLoaderModelOnly"]
    turbo = next(node for node in loras if node.get("widgets_values", [None])[0] == TURBO_LORA)
    creator = next(node for node in loras if node is not turbo)
    sigma = next(node for node in nodes if node["type"] == "MiniMaxH3SigmaShift")
    scheduler = next(node for node in nodes if node["type"] == "BasicScheduler")
    guider = next(node for node in nodes if node["type"] == "BasicGuider")
    sampler = next(node for node in nodes if node["type"] == "KSamplerSelect")
    actual = {(link_origin(link), link_target(link), link_type(link)) for link in links}
    expected = {
        (int(unet["id"]), int(turbo["id"]), "MODEL"),
        (int(turbo["id"]), int(creator["id"]), "MODEL"),
        (int(creator["id"]), int(sigma["id"]), "MODEL"),
        (int(sigma["id"]), int(scheduler["id"]), "MODEL"),
        (int(sigma["id"]), int(guider["id"]), "MODEL"),
    }
    if not expected <= actual:
        raise RuntimeError(f"Turbo model chain is incomplete: {sorted(expected - actual)}")
    if turbo.get("widgets_values") != [TURBO_LORA, 1.0]:
        raise RuntimeError("Turbo LoRA must remain fixed at strength 1.0")
    if sigma.get("widgets_values") != [12.0, 3.0]:
        raise RuntimeError("Turbo SigmaShift must remain video=12/audio=3")
    if scheduler.get("widgets_values") != ["simple", 8, 1]:
        raise RuntimeError("Turbo scheduler must remain simple at 8 steps")
    if sampler.get("widgets_values") != ["euler"]:
        raise RuntimeError("Turbo sampler must remain Euler")
    if any(node["type"] in {"EasyCache", "ApplyMiniMaxH3FirstBlockCache"} for node in nodes):
        raise RuntimeError("Turbo preset must not stack another approximate cache")


def graph_with_types(
    workflow: dict[str, object], required: set[str]
) -> dict[str, object]:
    graph = next(
        (
            candidate
            for candidate in graph_candidates(workflow)
            if required <= {
                str(node["type"]) for node in candidate.get("nodes", [])
            }
        ),
        None,
    )
    if graph is None:
        raise RuntimeError(
            f"No workflow graph contains required nodes: {', '.join(sorted(required))}"
        )
    return graph


def verify_story_wiring(
    workflow: dict[str, object], *, expect_easycache: bool, expect_auto_mosaic: bool
) -> None:
    required = {
        "UNETLoader",
        "LoraLoaderModelOnly",
        "MiniMaxH3OrderedStoryboard",
        "MiniMaxH3Director",
        "UpscaleModelLoader",
        "MiniMaxH3StoryExport2x",
    }
    if expect_easycache:
        required.add("EasyCache")
    graph = graph_with_types(workflow, required)
    nodes = graph["nodes"]
    links = graph["links"]
    by_type = {str(node["type"]): node for node in nodes}
    actual = {(link_origin(link), link_target(link), link_type(link)) for link in links}

    unet = by_type["UNETLoader"]
    lora = by_type["LoraLoaderModelOnly"]
    storyboard = by_type["MiniMaxH3OrderedStoryboard"]
    director = by_type["MiniMaxH3Director"]
    loader = by_type["UpscaleModelLoader"]
    exporter = by_type["MiniMaxH3StoryExport2x"]

    model_chain = {
        (int(unet["id"]), int(lora["id"]), "MODEL"),
    }
    if expect_easycache:
        cache = by_type["EasyCache"]
        model_chain |= {
            (int(lora["id"]), int(cache["id"]), "MODEL"),
            (int(cache["id"]), int(director["id"]), "MODEL"),
        }
        if cache.get("widgets_values") != [0.2, 0.15, 0.95, True]:
            raise RuntimeError("Story EasyCache fast defaults have changed")
    else:
        model_chain.add((int(lora["id"]), int(director["id"]), "MODEL"))

    image_chain = {
        (int(director["id"]), int(exporter["id"]), "IMAGE"),
    }
    if expect_auto_mosaic:
        mosaic = by_type["WanAutoMosaicVideo"]
        image_chain = {
            (int(director["id"]), int(mosaic["id"]), "IMAGE"),
            (int(mosaic["id"]), int(exporter["id"]), "IMAGE"),
        }
    data_chain = image_chain | {
        (int(storyboard["id"]), int(director["id"]), "MMX_DIR_GROUP"),
        (int(director["id"]), int(exporter["id"]), "AUDIO"),
        (int(director["id"]), int(exporter["id"]), "FLOAT"),
        (int(storyboard["id"]), int(exporter["id"]), "STRING"),
        (int(loader["id"]), int(exporter["id"]), "UPSCALE_MODEL"),
    }
    missing = (model_chain | data_chain) - actual
    if missing:
        raise RuntimeError(f"Ordered-story wiring is incomplete: {sorted(missing)}")

    direct_unet = (int(unet["id"]), int(director["id"]), "MODEL")
    direct_lora = (int(lora["id"]), int(director["id"]), "MODEL")
    if direct_unet in actual or (expect_easycache and direct_lora in actual):
        raise RuntimeError("The ordered-story model chain bypasses LoRA or EasyCache")

    timeline = None
    for value in director.get("widgets_values", []):
        if not isinstance(value, str) or not value.lstrip().startswith("{"):
            continue
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("output"), dict):
            timeline = candidate
            break
    if timeline is None:
        raise RuntimeError("Director timeline_data JSON is missing")
    output = timeline["output"]
    if output.get("exportMode") != "segments":
        raise RuntimeError("Long story Director must use per-segment export mode")
    if output.get("continuityEnabled") is not True:
        raise RuntimeError("Story Director Motion Context continuity must be enabled")
    if int(output.get("continuityOverlapFrames", 0)) != 22:
        raise RuntimeError("Story Director Motion Context must use 22 context frames")
    if float(timeline.get("frameRate", 0)) != 24.0:
        raise RuntimeError("Long story workflow must remain at 24 fps")

    storyboard_values = storyboard.get("widgets_values", ["{}"])
    storyboard_state = json.loads(str(storyboard_values[0]))
    if float(storyboard_state.get("defaults", {}).get("duration_sec", 0)) != 6.5:
        raise RuntimeError("Ordered Storyboard default duration must remain 6.5 seconds")
    if storyboard_state.get("images") != []:
        raise RuntimeError("Published Ordered Storyboard must not contain user input images")

    if loader.get("widgets_values") != ["RealESRGAN_x2plus.pth"]:
        raise RuntimeError("The story 2x upscaler model selection has changed")
    loader_models = loader.get("properties", {}).get("models", [])
    if len(loader_models) != 1 or loader_models[0].get("directory") != "upscale_models":
        raise RuntimeError("The story 2x upscaler model metadata is incomplete")

    exporter_values = exporter.get("widgets_values", [])
    if len(exporter_values) < 6:
        raise RuntimeError("Story Export 2x widgets are incomplete")
    if exporter_values[-2:] != [True, True]:
        raise RuntimeError("Story Export must remove boundary and loop duplicate frames")


def verify_auto_mosaic_wiring(workflow: dict[str, object], *, mode: str, expect_upscale: bool) -> None:
    mosaics = [node for node in all_nodes(workflow) if node["type"] == "WanAutoMosaicVideo"]
    if len(mosaics) != 1:
        raise RuntimeError(f"Auto-mosaic workflow must contain exactly one node, got {len(mosaics)}")
    mosaic = mosaics[0]
    if mosaic.get("widgets_values") != AUTO_MOSAIC_DEFAULTS:
        raise RuntimeError("Auto-mosaic defaults must remain JUST/0.30/0.50/0/3 without anus")
    if "anus" in str(mosaic.get("widgets_values", [""])[-1]).lower():
        raise RuntimeError("anus must not be present in default auto-mosaic targets")
    models = mosaic.get("properties", {}).get("models", [])
    if models != [
        {
            "name": "ntd11_anime_nsfw_segm_v5.pt",
            "url": "https://civitai.com/api/download/models/2266294",
            "directory": "auto_mosaic",
        }
    ]:
        raise RuntimeError("Auto-mosaic workflow model metadata is not pinned")

    graph = graph_with_types(workflow, {"WanAutoMosaicVideo"})
    nodes = graph["nodes"]
    links = graph["links"]
    actual = {(link_origin(link), link_target(link), link_type(link)) for link in links}
    target_type = "MiniMaxH3StoryExport2x" if mode == "story" else "CreateVideo"
    upstream_type = (
        "MiniMaxH3Director"
        if mode == "story"
        else "ImageUpscaleWithModel" if expect_upscale else "VAEDecode"
    )
    target = next(node for node in nodes if node["type"] == target_type)
    upstream = next(node for node in nodes if node["type"] == upstream_type)
    expected = {
        (int(upstream["id"]), int(mosaic["id"]), "IMAGE"),
        (int(mosaic["id"]), int(target["id"]), "IMAGE"),
    }
    if not expected <= actual:
        raise RuntimeError(f"Final-frame auto-mosaic wiring is incomplete: {sorted(expected - actual)}")
    if (int(upstream["id"]), int(target["id"]), "IMAGE") in actual:
        raise RuntimeError("Final video encoder bypasses auto mosaic")

    # The inserted node belongs to exactly one non-overlapping output group.
    mx, my = map(float, mosaic["pos"])
    mw, mh = map(float, mosaic["size"])
    containing = []
    groups = graph.get("groups", [])
    for group in groups:
        gx, gy, gw, gh = map(float, group["bounding"])
        if gx <= mx and gy <= my and mx + mw <= gx + gw and my + mh <= gy + gh:
            containing.append(group)
    if len(containing) != 1:
        raise RuntimeError("Auto-mosaic node must sit fully inside exactly one output group")
    for index, left in enumerate(groups):
        lx, ly, lw, lh = map(float, left["bounding"])
        for right in groups[index + 1 :]:
            rx, ry, rw, rh = map(float, right["bounding"])
            if max(lx, rx) < min(lx + lw, rx + rw) and max(ly, ry) < min(ly + lh, ry + rh):
                raise RuntimeError(
                    f"Workflow groups overlap: {left.get('title')} / {right.get('title')}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--mode", choices=("i2v", "r2v", "story"), required=True)
    parser.add_argument(
        "--custom-node-root",
        type=Path,
        action="append",
        default=[],
        help="Additional custom-node source tree used to verify non-core node types.",
    )
    parser.add_argument("--expect-easycache", action="store_true")
    parser.add_argument("--expect-first-block-cache", action="store_true")
    parser.add_argument("--expect-turbo", action="store_true")
    parser.add_argument("--expect-upscale", action="store_true")
    parser.add_argument("--expect-auto-mosaic", action="store_true")
    parser.add_argument("--auto-mosaic-manifest", type=Path)
    parser.add_argument(
        "--expect-lora",
        nargs="?",
        const=HMMOTION_LORA,
        choices=(HMMOTION_LORA, HMNSFW_V2_LORA),
    )
    parser.add_argument("--expect-lora-strength", type=float)
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
    if args.expect_auto_mosaic:
        if args.auto_mosaic_manifest is None:
            raise RuntimeError("--expect-auto-mosaic requires --auto-mosaic-manifest")
        mosaic_manifest = json.loads(args.auto_mosaic_manifest.read_text(encoding="utf-8"))
        provides = [
            str(path)
            for item in mosaic_manifest.get("files", [])
            for path in item.get("provides", [])
        ]
        if provides != [AUTO_MOSAIC_MODEL]:
            raise RuntimeError("Auto-mosaic manifest does not provide the pinned workflow model")
        expected_models.add(AUTO_MOSAIC_MODEL)
    if args.expect_turbo:
        expected_models.add(f"loras/{TURBO_LORA}")
    if args.expect_lora:
        expected_models.add(f"loras/{args.expect_lora}")
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
    elif args.mode == "r2v":
        if "MiniMaxH3ReferenceToVideo" not in node_types:
            raise RuntimeError("MiniMaxH3ReferenceToVideo is missing from the R2V workflow")
        if "MiniMaxH3ImageToVideo" in node_types or any("fl2va" in path for path in actual_models):
            raise RuntimeError("FL2VA assets are not allowed in an R2V workflow")
    else:
        required_story = {
            "MiniMaxH3OrderedStoryboard",
            "MiniMaxH3Director",
            "MiniMaxH3StoryExport2x",
        }
        if not required_story <= node_types:
            raise RuntimeError(
                f"Ordered-story nodes are missing: {sorted(required_story - node_types)}"
            )
        if "MiniMaxH3ReferenceToVideo" in node_types or any(
            "ref2va" in path for path in actual_models
        ):
            raise RuntimeError("Reference-to-video assets are not allowed in an I2V story workflow")

    if args.mode == "story":
        verify_story_wiring(
            workflow,
            expect_easycache=args.expect_easycache,
            expect_auto_mosaic=args.expect_auto_mosaic,
        )
    elif args.expect_easycache:
        verify_easycache_wiring(workflow)
    elif "EasyCache" in node_types:
        raise RuntimeError("EasyCache is enabled in a Quality workflow")
    if args.expect_first_block_cache:
        verify_first_block_cache(workflow)
    elif "ApplyMiniMaxH3FirstBlockCache" in node_types:
        raise RuntimeError("FirstBlockCache is enabled without an explicit Fast expectation")
    if args.expect_turbo:
        verify_turbo_8step(workflow)
    elif "MiniMaxH3SigmaShift" in node_types:
        raise RuntimeError("Turbo SigmaShift is enabled in a non-Turbo workflow")
    if args.require_video_reference:
        verify_video_reference_wiring(workflow)
    if args.mode == "story":
        pass
    elif args.expect_upscale:
        verify_upscale_wiring(
            workflow, expect_auto_mosaic=args.expect_auto_mosaic
        )
    elif {"UpscaleModelLoader", "ImageUpscaleWithModel"} & node_types:
        raise RuntimeError("Upscale nodes are enabled in a non-upscale workflow")
    if args.expect_lora and args.mode != "story":
        expected_strength = (
            args.expect_lora_strength
            if args.expect_lora_strength is not None
            else 1.0
        )
        verify_lora_wiring(workflow, args.expect_lora, expected_strength)
    elif not args.expect_lora and "LoraLoaderModelOnly" in node_types:
        raise RuntimeError("LoRA is enabled in a non-LoRA workflow")

    if args.expect_auto_mosaic:
        verify_auto_mosaic_wiring(
            workflow, mode=args.mode, expect_upscale=args.expect_upscale
        )
    elif "WanAutoMosaicVideo" in node_types:
        raise RuntimeError("WanAutoMosaicVideo is enabled in a normal workflow")

    if args.mode == "story":
        if not args.expect_upscale:
            raise RuntimeError("Ordered-story workflows must enable memory-bounded 2x export")
        if not args.expect_lora:
            raise RuntimeError("Ordered-story workflows must declare their selectable LoRA")
        graph = graph_with_types(workflow, {"LoraLoaderModelOnly"})
        lora = next(node for node in graph["nodes"] if node["type"] == "LoraLoaderModelOnly")
        expected_strength = (
            args.expect_lora_strength
            if args.expect_lora_strength is not None
            else 1.0
        )
        if lora.get("widgets_values") != [args.expect_lora, expected_strength]:
            raise RuntimeError(
                f"Story LoRA defaults changed; expected {args.expect_lora} at {expected_strength}"
            )
        model_entries = lora.get("properties", {}).get("models", [])
        if len(model_entries) != 1 or model_entries[0].get("directory") != "loras":
            raise RuntimeError("The story LoRA model metadata is incomplete")

    if args.comfyui_root:
        verify_comfyui_nodes(
            args.comfyui_root,
            node_types,
            subgraph_ids,
            args.custom_node_root,
        )

    native_count = len(node_types - subgraph_ids)
    speed = (
        "Turbo 8-step"
        if args.expect_turbo
        else "FirstBlockCache Fast"
        if args.expect_first_block_cache
        else "EasyCache Fast"
        if args.expect_easycache
        else "Quality"
    )
    output = " + Real-ESRGAN 2x" if args.expect_upscale else ""
    output += f" + {args.expect_lora}" if args.expect_lora else ""
    output += " + CPU Auto Mosaic" if args.expect_auto_mosaic else ""
    print(
        f"Verified {args.mode.upper()} {speed}{output} workflow: "
        f"{len(actual_models)} models, {native_count} native node types."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
