#!/usr/bin/env python3
"""Select one R2VA checkpoint AND its sampling settings; no extra node packs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = "04_MiniMax_H3_Character_R2V_2x.json"


def prepare(profile: str, project: Path = ROOT) -> tuple[dict, dict, dict]:
    profiles = json.loads((project / "manifests/r2v_profiles.json").read_text(encoding="utf-8"))["profiles"]
    if profile not in profiles:
        raise ValueError("H3_R2V_MODEL must be official, dasiwa-v2 or dasiwa-turbo-v2")
    settings = profiles[profile]
    manifest = json.loads((project / "manifests/minimax_h3_r2v_int8_upscale.json").read_text(encoding="utf-8"))
    workflow = json.loads((project / "workflows/character_reveal_r2v_int8_2x.json").read_text(encoding="utf-8"))
    if profile == "official":
        return manifest, copy.deepcopy(manifest), workflow

    # Replace, never append, the diffusion model. The four companion files stay pinned.
    manifest["files"] = [copy.deepcopy(settings["asset"]) if f["path"].startswith("diffusion_models/") else f
                         for f in manifest["files"]]
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"])
    manifest["r2v_profile"] = profile
    companions = copy.deepcopy(manifest)
    companions["files"] = [f for f in companions["files"] if f.get("auth") != "civitai"]
    companions["total_bytes"] = sum(f["size"] for f in companions["files"])

    by_type = {n["type"]: n for n in workflow["nodes"]}
    unet = by_type["UNETLoader"]
    asset = settings["asset"]
    name = Path(asset["path"]).name
    unet["widgets_values"] = [name, "default"]
    unet["title"] = f"MODEL · {settings['label']} (startup selected)"
    unet["properties"]["models"] = [{"name": name, "directory": "diffusion_models", "url": asset["source_url"]}]
    scheduler = by_type["BasicScheduler"]
    scheduler["widgets_values"] = [settings["scheduler"], settings["steps"], 1.0]
    scheduler["title"] = f"{settings['label']} · {settings['steps']} steps / simple"
    by_type["KSamplerSelect"]["widgets_values"] = [settings["sampler"]]

    # Both the scheduler and guider MUST receive the same shifted model.
    shift_id = max(n["id"] for n in workflow["nodes"]) + 1
    link_id = max(l[0] for l in workflow["links"]) + 1
    outgoing = [l for l in workflow["links"] if l[1] == unet["id"]]
    if {l[3] for l in outgoing} != {scheduler["id"], by_type["BasicGuider"]["id"]}:
        raise ValueError("Unexpected R2VA model consumers; refusing partial shift wiring")
    for link in outgoing:
        link[1], link[2] = shift_id, 0
    unet["outputs"][0]["links"] = [link_id]
    workflow["nodes"].append({
        "id": shift_id, "type": "MiniMaxH3SigmaShift", "title": "DaSiWa · Video / Audio shift (native)",
        "pos": [1280, 1000], "size": [540, 130], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": link_id}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [l[0] for l in outgoing]}],
        "properties": {"Node name for S&R": "MiniMaxH3SigmaShift"},
        "widgets_values": [settings["shift_video"], settings["shift_audio"]],
    })
    workflow["links"].append([link_id, unet["id"], 0, shift_id, 0, "MODEL"])
    pending, ordered, done = list(workflow["nodes"]), [], set()
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("R2VA graph contains a cycle")
        for node in ready:
            node["order"] = len(ordered)
            ordered.append(node)
            done.add(node["id"])
            pending.remove(node)
    workflow["nodes"] = ordered
    workflow["last_node_id"], workflow["last_link_id"] = shift_id, link_id
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"MiniMAXH-/r2v/{profile}/v1"))
    workflow["extra"]["character_r2v"].update(model_profile=profile, quality_tested_on_gpu=False)
    by_type["SaveVideo"]["widgets_values"][0] = f"video/MiniMax_H3_04_{profile}_R2VA_2x"
    for group in workflow["groups"]:
        if group["title"] == "Models · Ref2VA":
            group["title"] = "Models · DaSiWa Hybrid / R2VA"
    note = by_type["MarkdownNote"]
    note["widgets_values"] = [
        f"## 04 · {settings['label']} / R2VA 2x\n\n"
        "画像は外見参照。冒頭フレームに固定しません。同一キャラを1〜8枚追加できます。\n\n"
        "1. 左の『画像を追加』から1枚/複数枚をアップロード。削除・並べ替え可能。\n"
        "2. FULL PROMPTの1欄へ全文を貼る。音・BGMもここへ。4欄分割やモード切替は不要。\n"
        "<Picture 1>〜の番号は左の画像順。複数画像も同一キャラとして記述。\n"
        f"3. まず5秒・0.4MP・{settings['steps']}steps・seed固定で確認。\n\n"
        f"作者推奨範囲: {settings['sampler']} / simple、video shift {settings['shift_video']:g} / "
        f"audio shift {settings['shift_audio']:g}。旧FL2VA Turbo LoRAは重ねません。\n\n"
        "RefModは非圧縮Full Reference。画像ごとに縦横比を維持。長辺1024px・合計2048tokensを上限に縮小。"
        "INT8 VAE・生成モデル退避・タイルdecode・2x・CPUモザイクを維持。"
        "追加キャッシュ/LLM/API/構図画像は不要。画質改善やOOM回避の保証はありません。\n\n"
        f"起動設定 H3_R2V_MODEL={profile}。official / dasiwa-v2 / dasiwa-turbo-v2 から1つだけ取得。"
        "変更後は再起動し、Workflowsから04を開き直してください。旧キャンバスは自動更新されません。\n\n"
        "保存ノードのリンクからRefModをダウンロード可能。永続なしPodを削除する前に保存。\n\n"
        "非Turbo版は20–25steps、Turbo版は4/8stepsが作者の推奨。モデル名だけの交換はせず、"
        "対応する起動profileを使ってください。\n\n詳細: docs/dasiwa-r2v.md"
    ]
    return manifest, companions, workflow


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("H3_R2V_MODEL", "official"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, companions, workflow = prepare(args.profile)
    write_json(args.output_dir / "minimax_h3_r2v_models.json", manifest)
    write_json(args.output_dir / "minimax_h3_r2v_hf_models.json", companions)
    destination = args.output_dir / "workflows" / WORKFLOW
    if args.profile == "official":
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "workflows/character_reveal_r2v_int8_2x.json", destination)
    else:
        write_json(destination, workflow)
    print(f"[r2v-only] selected {args.profile}; {len(manifest['files'])} assets / {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
