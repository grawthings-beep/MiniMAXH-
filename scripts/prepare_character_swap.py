#!/usr/bin/env python3
"""Opt-in character replacement, preserving 04/05 and the selected base model."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import uuid

from prepare_r2v_profile import ROOT, prepare, write_json
from prepare_upscale_compare import merge_manifest

WORKFLOW = "06_MiniMax_H3_Character_Swap.json"
LORA = "h3_character_swap_pro4500_1000.safetensors"


def build(profile, project=ROOT, vae_profile="int8"):
    if profile not in {"official", "dasiwa-v2"}:
        raise ValueError("Character swap supports official / dasiwa-v2 only; speed-up profiles are not enabled.")
    manifest, _, workflow = prepare(profile, project, vae_profile=vae_profile)
    stock, _, stock_workflow = prepare(profile, project, vae_profile="int8")
    stock_vae = next(f for f in stock["files"] if f["path"].startswith("vae/minimax_h3_video_vae"))
    addon = json.loads((project / "manifests/character_swap.json").read_text(encoding="utf-8"))
    guard_assets = json.loads((project / "manifests/swap_guard.json").read_text(encoding="utf-8"))
    manifest = merge_manifest(manifest, {"files": [stock_vae]}, addon, guard_assets)
    manifest["character_swap"] = True
    nodes = {n["type"]: n for n in workflow["nodes"]}
    removed = {nodes[t]["id"] for t in ("ResolutionSelector", "MiniMaxH3FullPrompt", "MiniMaxH3CharacterRefModR2V")}
    workflow["nodes"] = [n for n in workflow["nodes"] if n["id"] not in removed]
    workflow["links"] = [l for l in workflow["links"] if l[1] not in removed and l[3] not in removed]
    for node in workflow["nodes"]:
        node["pos"][0] += 640
    for group in workflow["groups"]:
        group["bounding"][0] += 640
    workflow["groups"].insert(0, {"id": 8, "title": "01 · 元動画 / 必要区間だけ読み込み", "bounding": [10, 0, 600, 1810],
                                 "color": "#3f789e", "font_size": 24, "flags": {}})

    def add(kind, pos, size, inputs=(), outputs=(), widgets=(), title=None):
        node = {"id": max(n["id"] for n in workflow["nodes"]) + 1, "type": kind,
                "title": title or kind, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": 0,
                "inputs": [{"name": name, "type": typ, "link": None} for name, typ in inputs],
                "outputs": [{"name": name, "type": typ, "links": []} for name, typ in outputs],
                "widgets_values": list(widgets), "properties": {"Node name for S&R": kind}}
        workflow["nodes"].append(node)
        return node

    def link(source, slot, target, target_slot, typ):
        workflow["links"].append([max(l[0] for l in workflow["links"]) + 1, source["id"], slot, target["id"], target_slot, typ])

    loader = add("LoadVideo", [40, 80], [540, 640], outputs=(("VIDEO", "VIDEO"),), widgets=[""], title="元動画をアップロード")
    clip = add("MiniMaxH3SwapClip", [40, 800], [540, 230], inputs=(("video", "VIDEO"),),
               outputs=(("source", "H3_SWAP_CLIP"), ("actual_clip", "STRING")), widgets=[0., 5.2, .4])
    preview = add("PreviewAny", [40, 1110], [540, 250], inputs=(("source", "*"),), outputs=(("STRING", "STRING"),),
                  title="実際の生成尺・解像度（実行後）")
    spec = importlib.util.spec_from_file_location("swap_defaults", project / "custom_nodes/minimax_h3_ordered_storyboard/swap_nodes.py")
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)
    prompt = add("MiniMaxH3SwapPrompt", [1200, 80], [620, 1100], outputs=(("STRING", "STRING"),), widgets=[defaults.GUARDED_PROMPT])
    conditioning = add("MiniMaxH3CharacterSwap", [2560, 80], [480, 400],
                       inputs=(("clip", "CLIP"), ("character", "H3_CHARACTER_REFMOD"), ("source", "H3_SWAP_CLIP"),
                               ("vae", "VAE"), ("prompt", "STRING"), ("edit", "H3_SWAP_EDIT")),
                       outputs=(("positive", "CONDITIONING"), ("LATENT", "LATENT")), widgets=[""],
                       title="MASK REVIEW · preview → paste approval code → generate")
    audio = add("MiniMaxH3SwapAudio", [3660, 1250], [500, 140], inputs=(("source", "H3_SWAP_CLIP"), ("generated", "AUDIO")),
                outputs=(("AUDIO", "AUDIO"),), widgets=["source"])
    nodes["CreateVideo"]["pos"] = [3660, 1480]
    video_vae = next(n for n in workflow["nodes"] if n["type"] == "VAELoader" and n["id"] == 119)
    if vae_profile == "x2-detail":
        ref_vae = copy.deepcopy(next(n for n in stock_workflow["nodes"] if n["id"] == 119))
        ref_vae.update(id=max(n["id"] for n in workflow["nodes"])+1, pos=[1920, 1530],
                       title="参照専用INT8 VAE · 画像/元動画のencode")
        workflow["nodes"].append(ref_vae)
    else:
        ref_vae = video_vae
    refs = nodes["MiniMaxH3CreateCharacterRefMod"]
    for connection in workflow["links"]:
        if connection[3:5] == [refs["id"], 1]:
            connection[1:3] = [ref_vae["id"], 0]
        if connection[1] == nodes["VAEDecodeAudio"]["id"] and connection[3] == nodes["CreateVideo"]["id"]:
            connection[1:3] = [audio["id"], 0]
    for a, ai, b, bi, typ in (
        (loader, 0, clip, 0, "VIDEO"), (clip, 1, preview, 0, "STRING"),
        (nodes["CLIPLoader"], 0, conditioning, 0, "CLIP"), (refs, 0, conditioning, 1, "H3_CHARACTER_REFMOD"),
        (clip, 0, conditioning, 2, "H3_SWAP_CLIP"), (ref_vae, 0, conditioning, 3, "VAE"),
        (prompt, 0, conditioning, 4, "STRING"), (conditioning, 0, nodes["BasicGuider"], 1, "CONDITIONING"),
        (conditioning, 1, nodes["SamplerCustomAdvanced"], 4, "LATENT"),
        (clip, 0, audio, 0, "H3_SWAP_CLIP"), (nodes["VAEDecodeAudio"], 0, audio, 1, "AUDIO"),
    ):
        link(a, ai, b, bi, typ)
    target = add("MiniMaxH3SwapTarget", [40, 1450], [540, 1100], inputs=(("source", "H3_SWAP_CLIP"), ("target_mask", "MASK")),
                 outputs=(("edit", "H3_SWAP_EDIT"), ("mask_report", "STRING")),
                 widgets=["masked_replace", .25, 32, .1, 48], title="CPU人物範囲 · 全フレームを確認 / originalで従来方式")
    target["inputs"][1]["shape"] = 7
    asset = guard_assets["files"][0]
    target["properties"]["models"] = [{"name": "yolo11s-seg.pt", "directory": "swap_detection", "url": asset["source_url"]}]
    mask_report = add("PreviewAny", [2560, 600], [480, 520], inputs=(("source", "*"),), outputs=(("STRING", "STRING"),),
                      title="マスク検出結果 / approval code（未検出＝退場とは限りません）")
    finish = add("MiniMaxH3SwapFinish", [3660, 1000], [500, 280],
                 inputs=(("images", "IMAGE"), ("source", "H3_SWAP_CLIP"), ("edit", "H3_SWAP_EDIT")),
                 outputs=(("images", "IMAGE"), ("review_report", "STRING")), widgets=[True, .985],
                 title="元人物のコピーを拒否 · 背景合成 / 正しいキャラかは目視")
    review = add("PreviewAny", [4260, 830], [540, 300], inputs=(("source", "*"),), outputs=(("STRING", "STRING"),),
                 title="出力検査 / 置換の正しさを保証する判定ではありません")
    mosaic = nodes["WanAutoMosaicVideo"]
    incoming = next(l for l in workflow["links"] if l[3:5] == [mosaic["id"], 0])
    upstream = next(n for n in workflow["nodes"] if n["id"] == incoming[1])
    upstream_slot = incoming[2]
    incoming[1:3] = [finish["id"], 0]
    for a, ai, b, bi, typ in (
        (clip, 0, target, 0, "H3_SWAP_CLIP"), (target, 0, conditioning, 5, "H3_SWAP_EDIT"),
        (target, 1, mask_report, 0, "STRING"), (upstream, upstream_slot, finish, 0, "IMAGE"),
        (clip, 0, finish, 1, "H3_SWAP_CLIP"), (target, 0, finish, 2, "H3_SWAP_EDIT"),
        (finish, 1, review, 0, "STRING"),
    ):
        link(a, ai, b, bi, typ)
    for group in workflow["groups"]:
        group["bounding"][3] = 2640
    nodes["VAEDecodeAudio"]["pos"] = [3660, 650]
    if vae_profile == "int8":
        nodes["ImageUpscaleWithModel"]["pos"] = [3660, 820]
    mosaic["pos"] = [3660, 1380]
    audio["pos"] = [3660, 1900]
    nodes["CreateVideo"]["pos"] = [3660, 2120]
    lora = nodes["MiniMaxH3R2VLoRA"]
    lora.update(title="人物置換LoRA · 選択/強度/ON・OFF", widgets_values=[LORA, 1., True])
    lora["properties"]["models"] = [{"name": LORA, "directory": "loras", "url": addon["files"][0]["source_url"]}]
    nodes["BasicScheduler"].update(widgets_values=["simple", 20, 1.0], title="人物置換 · 20 steps / simple")
    nodes["KSamplerSelect"]["widgets_values"] = ["res_multistep"]
    nodes["SaveVideo"]["widgets_values"][0] = "video/MiniMax_H3_06_Character_Swap"
    nodes["MarkdownNote"]["widgets_values"] = [
        "## 06 · 人物置換 / Masked + copy guard\n\n初回はCPUマスク確認で停止。全フレームを確認し、"
        "検出結果のapproval codeを右のmask_approvalへ貼って再実行。未検出フレームは退場か検出漏れかを確認。"
        "複数人物は自動選択せず停止。必要なら整列済みMASKを接続。\n\n"
        "mode=originalは従来の生成経路へ戻ります（旧プロンプトはdocs/character-swap.md）。"
        "masked_replaceは人物部分の元外見をぼかして条件へ渡し、対象セルだけ再生成。"
        "参照画像は1〜8枚、顔の大きい1枚から。"
        "初期0.4MP / 約5秒 / 20steps。人物置換LoRA 1.0のみON（OFF比較可能）。\n\n"
        "元動画を24fpsへ時間基準で変換し、末尾を17n+5フレームへ切り詰めます。速度は変更しません。"
        "5秒ちょうどの動画は107frames / 4.458秒。5.2秒以上なら124frames / 5.167秒。\n\n"
        "<Video 1> = 元動画、<Picture 1>〜 = 左のキャラ画像順。複数人物なら服や位置で対象を明記。"
        "元動画にカットがある場合は区間を分けてください。\n\n"
        "音声: source=元動画（無音動画は無音）、generated=生成、mute=無音。"
        "元音声はトリミング後に再エンコード。口形との一致は保証しません。\n\n"
        "DaSiWaは既定で維持。元人物のコピー検査は正しい参照キャラの証明ではありません。"
        "新masked方式のGPU置換品質は未検証。"
        "画像だけよりVRAMが必要。背景/動き/人物の完全保持やOOM回避は保証しません。"
        "既存04/05は変更なし。docs/character-swap.md"
    ]
    by_id = {n["id"]: n for n in workflow["nodes"]}
    for n in workflow["nodes"]:
        for i in n.get("inputs", []):
            i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
    for lid, a, ai, b, bi, _ in workflow["links"]:
        by_id[a]["outputs"][ai]["links"].append(lid)
        by_id[b]["inputs"][bi]["link"] = lid
    pending, done, ordered = list(workflow["nodes"]), set(), []
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("Character swap graph contains a cycle")
        for n in ready:
            n["order"] = len(ordered)
            ordered.append(n)
            done.add(n["id"])
            pending.remove(n)
    workflow["nodes"] = ordered
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"MiniMAXH-/swap/{profile}/{vae_profile}/v2-guarded"))
    workflow["last_node_id"] = max(by_id)
    workflow["last_link_id"] = max(l[0] for l in workflow["links"])
    workflow["extra"]["character_r2v"].update(character_swap=True, reference_vae="int8", quality_tested_on_gpu=False,
                                             swap_revision=2, mask_review_required=True, copy_check_is_identity_check=False)
    return manifest, workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", default="shared", choices=("shared", "official"))
    parser.add_argument("--vae", default="int8", choices=("int8", "x2-detail"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = args.profile if args.model == "shared" else "official"
    manifest, workflow = build(selected, vae_profile=args.vae)
    from verify_character_swap import verify
    verify(workflow, manifest)
    normal_path = args.output_dir / "minimax_h3_r2v_models.json"
    # May already include comparison assets. Keep 04's exact validator input separate.
    current = json.loads(normal_path.read_text(encoding="utf-8"))
    normal = prepare(args.profile, vae_profile=args.vae)[0]
    write_json(args.output_dir / "minimax_h3_r2v_04_models.json", normal)
    merged = merge_manifest(current, manifest)
    write_json(normal_path, merged)
    hf = copy.deepcopy(merged)
    hf["files"] = [f for f in hf["files"] if f.get("auth") != "civitai"]
    hf["total_bytes"] = sum(f["size"] for f in hf["files"])
    write_json(args.output_dir / "minimax_h3_r2v_hf_models.json", hf)
    write_json(args.output_dir / "minimax_h3_swap_models.json", manifest)
    write_json(args.output_dir / "workflows" / WORKFLOW, workflow)
    print(f"[h3-swap] installed guarded 06; base={selected}; {len(merged['files'])} unique assets / {merged['total_bytes']} bytes; mask review required")


if __name__ == "__main__":
    main()
