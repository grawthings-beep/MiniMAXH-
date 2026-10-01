#!/usr/bin/env python3
"""Opt-in finishing comparison, with one INT8-reference sampling path."""
from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare, write_json

WORKFLOW = "05_MiniMax_H3_Upscale_Compare.json"


def merge_manifest(base, *others):
    result = copy.deepcopy(base)
    assets = {f["path"]: f for f in result["files"]}
    for other in others:
        for asset in other["files"]:
            if asset["path"] in assets and assets[asset["path"]] != asset:
                raise ValueError(f"Conflicting model definitions: {asset['path']}")
            assets[asset["path"]] = copy.deepcopy(asset)
    result["files"] = list(assets.values())
    result["total_bytes"] = sum(f["size"] for f in result["files"])
    return result


def build(profile, project=ROOT):
    manifest, _, workflow = prepare(profile, project, vae_profile="int8")
    manifest["files"] = [f for f in manifest["files"] if not f["path"].startswith("upscale_models/")]
    extra = [json.loads((project / f"manifests/{name}.json").read_text(encoding="utf-8"))
             for name in ("x2_detail_vae", "upscale_compare")]
    manifest = merge_manifest(manifest, *extra)
    manifest["comparison"] = True
    nodes = {n["type"]: n for n in workflow["nodes"]}
    removed = {nodes[t]["id"] for t in ("UpscaleModelLoader", "ImageUpscaleWithModel", "MiniMaxH3VAEDecodeTiled",
                                        "WanAutoMosaicVideo", "CreateVideo", "SaveVideo")}
    workflow["nodes"] = [n for n in workflow["nodes"] if n["id"] not in removed]
    workflow["links"] = [l for l in workflow["links"] if l[1] not in removed and l[3] not in removed]
    video_vae = next(n for n in workflow["nodes"] if n["type"] == "VAELoader"
                     and "video_vae" in n["widgets_values"][0])
    compare_id = max(n["id"] for n in workflow["nodes"]) + 1
    compare = {
        "id": compare_id, "type": "MiniMaxH3CompareUpscale", "title": "05 · 同一latent / A・B・C比較保存",
        "pos": [3620, 80], "size": [1200, 1350], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": name, "type": typ, "link": None, **({"widget": {"name": name}} if name in {"width", "height"} else {})}
                   for name, typ in (("samples", "LATENT"), ("standard_vae", "VAE"), ("audio", "AUDIO"),
                                     ("width", "INT"), ("height", "INT"))],
        "outputs": [{"name": "comparison_report", "type": "STRING", "links": []}],
        "widgets_values": [480, 864, 24.0, "video/H3_Upscale_Compare", True],
        "properties": {"Node name for S&R": "MiniMaxH3CompareUpscale", "models": [
            {"name": Path(f["path"]).name, "directory": str(Path(f["path"]).parent).replace("\\", "/"), "url": f["source_url"]}
            for config in extra for f in config["files"]
        ] + copy.deepcopy(nodes["WanAutoMosaicVideo"]["properties"]["models"])},
    }
    workflow["nodes"].append(compare)
    for source, slot, target_slot, typ in (
        (nodes["MiniMaxH3ReleaseVRAMLatent"], 0, 0, "LATENT"), (video_vae, 0, 1, "VAE"),
        (nodes["VAEDecodeAudio"], 0, 2, "AUDIO"), (nodes["ResolutionSelector"], 0, 3, "INT"),
        (nodes["ResolutionSelector"], 1, 4, "INT"),
    ):
        workflow["links"].append([max(l[0] for l in workflow["links"]) + 1, source["id"], slot, compare_id, target_slot, typ])
    by_id = {n["id"]: n for n in workflow["nodes"]}
    for node in workflow["nodes"]:
        for socket in node.get("inputs", []):
            socket["link"] = None
        for socket in node.get("outputs", []):
            socket["links"] = []
    for lid, source, slot, target, target_slot, _ in workflow["links"]:
        by_id[source]["outputs"][slot]["links"].append(lid)
        by_id[target]["inputs"][target_slot]["link"] = lid
    pending, done, ordered = list(workflow["nodes"]), set(), []
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("Comparison graph cycle")
        for n in ready:
            n["order"] = len(ordered)
            done.add(n["id"])
            ordered.append(n)
            pending.remove(n)
    workflow["nodes"] = ordered
    nodes["VAEDecodeAudio"]["pos"] = [3020, 250]
    workflow["groups"][-2]["title"] = "06 · H3退避 / 共通音声"
    workflow["groups"][-1]["title"] = "07 · 仕上げ比較 / 3本を順番に保存"
    workflow["groups"][-1]["bounding"][2] = 1260
    nodes["MarkdownNote"]["widgets_values"] = [
        "## 05 · 仕上げ方式の比較\n\n"
        "同じ生成latent・音声から A: X2 VAE / B: INT8 VAE + SPAN / C: INT8 VAE + AnimeSharp を保存。"
        "すべて縦横2倍・24fps・H264 CRF18。生成は1回。\n\n"
        "参照画像のエンコードはINT8固定。04のX2参照エンコードと全工程を比較するものではありません。\n\n"
        "右の比較ノードでモザイクON/OFF。標準はCPU JUST輪郭。"
        "3本を順番に処理し、処理時間JSONも保存。CPU RAMには完成フレームが残ります。"
        "まず5秒・0.4MP、seed固定。OOM回避・画質優位は未保証。\n\n"
        "AnimeSharp: Kim2091 / CC-BY-NC-SA-4.0（非商用）。SPAN: Helaman / CC-BY-4.0。"
        "通常04は変更なし。詳細: docs/upscale-compare.md"
    ]
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"MiniMAXH-/r2v/{profile}/compare-v1"))
    workflow["last_node_id"] = max(by_id)
    workflow["last_link_id"] = max(l[0] for l in workflow["links"])
    workflow["extra"]["character_r2v"].update(comparison=True, reference_vae="int8", quality_tested_on_gpu=False)
    return manifest, workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, workflow = build(args.profile)
    from verify_upscale_compare import verify
    verify(workflow, manifest)
    # Preserve the exact 04 manifest for its existing validator. Merge only downloads.
    normal_path = args.output_dir / "minimax_h3_r2v_models.json"
    normal = json.loads(normal_path.read_text(encoding="utf-8"))
    write_json(args.output_dir / "minimax_h3_r2v_04_models.json", normal)
    merged = merge_manifest(normal, manifest)
    write_json(normal_path, merged)
    hf = copy.deepcopy(merged)
    hf["files"] = [f for f in hf["files"] if f.get("auth") != "civitai"]
    hf["total_bytes"] = sum(f["size"] for f in hf["files"])
    write_json(args.output_dir / "minimax_h3_r2v_hf_models.json", hf)
    write_json(args.output_dir / "minimax_h3_compare_models.json", manifest)
    write_json(args.output_dir / "workflows" / WORKFLOW, workflow)
    print(f"[h3-compare] installed 05; {len(merged['files'])} unique assets / {merged['total_bytes']} bytes")


if __name__ == "__main__":
    main()
