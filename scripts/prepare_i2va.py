#!/usr/bin/env python3
"""Add a first-frame I2VA preset beside the existing character-reference R2VA 04."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import uuid
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare, write_json
from prepare_t2va import build as build_t2va
from prepare_upscale_compare import merge_manifest

WORKFLOW = "09_MiniMax_H3_I2VA_First_Frame_2x.json"


def _replace_type(node: dict, kind: str, title: str) -> None:
    node.update(type=kind, title=title)
    node["properties"]["Node name for S&R"] = kind


def build(profile: str, project: Path = ROOT) -> tuple[dict, dict]:
    if profile != "dasiwa-v2":
        raise ValueError("H3_I2VA requires H3_R2V_MODEL=dasiwa-v2 (shared FL2VA-capable checkpoint)")
    _, workflow = build_t2va(profile, project)
    manifest, _, baseline = prepare(profile, project, vae_profile="int8")
    workflow = copy.deepcopy(workflow)
    nodes = workflow["nodes"]
    by_type = {n["type"]: n for n in nodes}

    spec = importlib.util.spec_from_file_location("i2va_defaults", project / "custom_nodes/minimax_h3_ordered_storyboard/t2va_nodes.py")
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)

    prompt = by_type["MiniMaxH3T2VAPrompt"]
    _replace_type(prompt, "MiniMaxH3I2VAPrompt", "01 · 画像から開始 / FULL PROMPT + 尺")
    prompt["widgets_values"] = [5.0, defaults.I2VA_DEFAULT_PROMPT]
    cond = by_type["MiniMaxH3TextToVideo"]
    _replace_type(cond, "MiniMaxH3FirstFrameToVideo", "I2VA · 画像そのものが0秒のフレーム")
    old_inputs = {item["name"]: i for i, item in enumerate(cond["inputs"])}
    new_inputs = [
        {"name": name, "type": kind, "link": None}
        for name, kind in (("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"),
                           ("prompt", "STRING"), ("width", "INT"), ("height", "INT"), ("length", "INT"))
    ]
    for item in new_inputs:
        if item["name"] in {"width", "height", "length"}:
            item["widget"] = {"name": item["name"]}
    cond["inputs"] = new_inputs
    for link in workflow["links"]:
        if link[3] == cond["id"]:
            original_name = next(name for name, slot in old_inputs.items() if slot == link[4])
            link[4] = next(i for i, item in enumerate(new_inputs) if item["name"] == original_name)

    first_frame = {
        "id": max(n["id"] for n in nodes) + 1, "type": "LoadImage", "title": "最初の画像 · ここにアップロード",
        "pos": [40, 80], "size": [620, 500], "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []},
                    {"name": "MASK", "type": "MASK", "links": []}],
        "properties": {"Node name for S&R": "LoadImage"}, "widgets_values": ["", "image"],
    }
    nodes.append(first_frame)
    video_vae = next(n for n in nodes if n["type"] == "VAELoader" and "video_vae" in n["widgets_values"][0])
    for source, out_slot, name, kind in ((first_frame, 0, "first_frame", "IMAGE"),
                                        (video_vae, 0, "vae", "VAE")):
        workflow["links"].append([max(l[0] for l in workflow["links"]) + 1,
                                  source["id"], out_slot, cond["id"],
                                  next(i for i, item in enumerate(cond["inputs"]) if item["name"] == name), kind])

    # The shared INT8 VAE is required to encode the opening frame. Upscale the
    # completed video after decode, never the input still or generated latent.
    baseline_types = {n["type"]: n for n in baseline["nodes"]}
    up_model = copy.deepcopy(baseline_types["UpscaleModelLoader"])
    upscaler = copy.deepcopy(baseline_types["ImageUpscaleWithModel"])
    original_ids = {n["id"] for n in nodes}
    up_model["id"] = max(original_ids) + 1
    upscaler["id"] = up_model["id"] + 1
    nodes.extend((up_model, upscaler))
    mosaic = by_type["WanAutoMosaicVideo"]
    decode = by_type["MiniMaxH3VAEDecodeTiled"]
    workflow["links"] = [l for l in workflow["links"] if not (l[1] == decode["id"] and l[3] == mosaic["id"])]
    for source, out_slot, target, in_name, kind in (
        (up_model, 0, upscaler, "upscale_model", "UPSCALE_MODEL"),
        (decode, 0, upscaler, "image", "IMAGE"),
        (upscaler, 0, mosaic, "images", "IMAGE"),
    ):
        workflow["links"].append([max(l[0] for l in workflow["links"]) + 1,
                                  source["id"], out_slot, target["id"],
                                  next(i for i, item in enumerate(target["inputs"]) if item["name"] == in_name), kind])

    by_type["PreviewAny"]["title"] = "実際の尺・解像度（実行後）"
    by_type["MiniMaxH3T2VAResolution"]["title"] = "元動画の縦横比 / 0.4MP推奨"
    by_type["SaveVideo"]["widgets_values"][0] = "video/MiniMax_H3_09_I2VA_First_Frame_2x"
    by_type["SaveVideo"]["title"] = "完成MP4 · 2x / 音声付き"
    by_type["MarkdownNote"]["widgets_values"] = [
        "## 09 · I2VA / 画像そのものから始まる動画\n\n"
        "左上の『最初の画像』をアップロード。この画像が0秒の画面になります。"
        "04 REF2VAとは違い、冒頭を別構図から始める用途には使いません。\n\n"
        "FULL PROMPTへ映像・効果音・BGMをまとめて入力。"
        "まず5秒、0.4MP、seed固定で確認。"
        "元画像の画角と生成比率が大きく異なるとクロップ/変形が起こり得ます。\n\n"
        "DaSiWa Hybrid v2 INT8を04と共有。25step/simple/shift 11,4。"
        "INT8 VAEでデコード→RealESRGAN 2x→CPU自動モザイク→MP4。"
        "入力画像へモザイクは入れません。LoRA/RefModは09に適用しません。\n\n"
        "画像のキャラだけを参照し、違う画面から始めたい場合は04 REF2VAを使用。"
        "09を表示するにはH3_PROFILE=r2v / H3_R2V_MODEL=dasiwa-v2 / H3_I2VA=1。"
        "GPUでの画質・OOMは未検証。"
    ]

    layout = {
        "LoadImage": (40, 80, 620, 500), "MiniMaxH3I2VAPrompt": (40, 670, 620, 780),
        "MiniMaxH3T2VAResolution": (40, 1540, 620, 180), "MarkdownNote": (40, 1800, 620, 560),
        "UpscaleModelLoader": (1920, 660, 500, 100),
        "ImageUpscaleWithModel": (1920, 840, 500, 140),
        "WanAutoMosaicVideo": (1920, 1080, 500, 480),
        "CreateVideo": (1920, 1660, 500, 140),
    }
    for node in nodes:
        if node["type"] in layout:
            x, y, width, height = layout[node["type"]]
            node["pos"], node["size"] = [x, y], [width, height]
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for group in workflow["groups"]:
        group["bounding"][3] = 2450
    workflow["groups"][0]["title"] = "01 · First frame / Prompt / Resolution"
    workflow["groups"][3]["title"] = "04 · Decode / RealESRGAN 2x / CPU Mosaic"
    workflow["groups"][4]["title"] = "05 · 完成I2VA MP4"
    by_id = {n["id"]: n for n in nodes}
    for lid, origin, out_slot, target, in_slot, _ in workflow["links"]:
        by_id[origin]["outputs"][out_slot]["links"].append(lid)
        by_id[target]["inputs"][in_slot]["link"] = lid
    pending, ordered, done = list(nodes), [], set()
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("I2VA graph contains a cycle")
        for node in ready:
            node["order"] = len(ordered)
            ordered.append(node)
            done.add(node["id"])
            pending.remove(node)
    workflow["nodes"] = ordered
    workflow["last_node_id"] = max(by_id)
    workflow["last_link_id"] = max(l[0] for l in workflow["links"])
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "MiniMAXH-/i2va-first-frame-dasiwa-v2-v1"))
    workflow["extra"].pop("t2va", None)
    workflow["extra"]["i2va"] = {"model_profile": profile, "first_frame_required": True,
                                  "upscale": "RealESRGAN_x2plus", "gpu_generation_tested": False}
    manifest["i2va"] = True
    return manifest, workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vae", choices=("int8", "x2-detail"), default="int8",
                        help="04 R2VA VAE profile; I2VA always uses INT8 encode/decode")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, workflow = build(args.profile)
    from verify_i2va import verify
    verify(workflow, manifest)
    target = args.output_dir / "minimax_h3_r2v_models.json"
    merged = merge_manifest(json.loads(target.read_text(encoding="utf-8")), manifest)
    write_json(args.output_dir / "minimax_h3_r2v_04_models.json", prepare(args.profile, vae_profile=args.vae)[0])
    write_json(target, merged)
    hf = copy.deepcopy(merged)
    hf["files"] = [item for item in hf["files"] if item.get("auth") != "civitai"]
    hf["total_bytes"] = sum(item["size"] for item in hf["files"])
    write_json(args.output_dir / "minimax_h3_r2v_hf_models.json", hf)
    write_json(args.output_dir / "minimax_h3_i2va_models.json", manifest)
    write_json(args.output_dir / "workflows" / WORKFLOW, workflow)
    print("[h3-i2va] installed 09: first-frame image, shared DaSiWa v2, RealESRGAN 2x, CPU mosaic")


if __name__ == "__main__":
    main()
