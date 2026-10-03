#!/usr/bin/env python3
"""Opt-in text-only motion masters; share DaSiWa, leave 04/05/06 unchanged."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import uuid

from prepare_r2v_profile import ROOT, prepare, write_json
from prepare_upscale_compare import merge_manifest

WORKFLOW = "07_MiniMax_H3_Text_to_Video.json"


def build(profile, project=ROOT):
    # 'official' in this repo is Ref2VA, NOT the official FL2VA/T2VA checkpoint.
    if profile != "dasiwa-v2":
        raise ValueError("H3_T2VA requires H3_R2V_MODEL=dasiwa-v2; official is Ref2VA, not FL2VA.")
    manifest, _, baseline = prepare(profile, project, vae_profile="int8")
    keep = {"UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3SigmaShift", "BasicScheduler",
            "KSamplerSelect", "RandomNoise", "BasicGuider", "SamplerCustomAdvanced",
            "MiniMaxH3ReleaseVRAMLatent", "MiniMaxH3VAEDecodeTiled", "VAEDecodeAudio",
            "WanAutoMosaicVideo", "CreateVideo", "SaveVideo", "ResolutionSelector", "MarkdownNote"}
    nodes = [copy.deepcopy(n) for n in baseline["nodes"] if n["type"] in keep]
    resolution = next(n for n in nodes if n["type"] == "ResolutionSelector")
    resolution.update(type="MiniMaxH3T2VAResolution", properties={"Node name for S&R": "MiniMaxH3T2VAResolution"})
    workflow = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "MiniMAXH-/text-only-v1")), "revision": 0,
                "version": 0.4, "nodes": nodes, "links": [], "groups": [], "config": {},
                "extra": {"ds": {"scale": 0.65, "offset": [30, 30]},
                          "t2va": {"model_profile": profile, "reference_inputs": 0,
                                   "upscale": False, "quality_tested_on_gpu": False}}}
    by_type = {n["type"]: n for n in nodes}
    video_vae = next(n for n in nodes if n["type"] == "VAELoader" and "video_vae" in n["widgets_values"][0])
    audio_vae = next(n for n in nodes if n["type"] == "VAELoader" and "audio_vae" in n["widgets_values"][0])
    ids = {n["id"] for n in nodes}
    workflow["links"] = [l for l in baseline["links"] if l[1] in ids and l[3] in ids]

    def add(kind, inputs=(), outputs=(), widgets=(), title=None):
        n = {"id": max(n["id"] for n in nodes) + 1, "type": kind, "title": title or kind,
             "flags": {}, "mode": 0, "order": 0, "properties": {"Node name for S&R": kind},
             "widgets_values": list(widgets),
             "inputs": [{"name": name, "type": typ, "link": None,
                         **({"widget": {"name": name}} if typ == "INT" else {})} for name, typ in inputs],
             "outputs": [{"name": name, "type": typ, "links": []} for name, typ in outputs]}
        nodes.append(n)
        by_type[kind] = n
        return n

    spec = importlib.util.spec_from_file_location("t2va_defaults", project / "custom_nodes/minimax_h3_ordered_storyboard/t2va_nodes.py")
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)
    prompt = add("MiniMaxH3T2VAPrompt", outputs=(("prompt", "STRING"), ("length", "INT")),
                 widgets=(5., defaults.DEFAULT_PROMPT), title="01 · 画像なし / FULL PROMPT + 尺")
    cond = add("MiniMaxH3TextToVideo", inputs=(("clip", "CLIP"), ("prompt", "STRING"),
               ("width", "INT"), ("height", "INT"), ("length", "INT")),
               outputs=(("positive", "CONDITIONING"), ("LATENT", "LATENT"), ("actual_clip", "STRING")),
               widgets=(448, 832, 124), title="T2VA · 参照ゼロ / 冒頭構図は文章で指定")
    report = add("PreviewAny", inputs=(("source", "*"),), outputs=(("STRING", "STRING"),), title="実際の尺・解像度（実行後）")

    def link(a, out, b, inp):
        ai = next(i for i, s in enumerate(a["outputs"]) if s["name"] == out)
        bi = next(i for i, s in enumerate(b["inputs"]) if s["name"] == inp)
        workflow["links"].append([max((l[0] for l in workflow["links"]), default=0)+1,
                                  a["id"], ai, b["id"], bi, a["outputs"][ai]["type"]])

    for a, out, b, inp in (
        ("UNETLoader", "MODEL", "MiniMaxH3SigmaShift", "model"),
        ("CLIPLoader", "CLIP", "MiniMaxH3TextToVideo", "clip"),
        ("MiniMaxH3T2VAPrompt", "prompt", "MiniMaxH3TextToVideo", "prompt"),
        ("MiniMaxH3T2VAPrompt", "length", "MiniMaxH3TextToVideo", "length"),
        ("MiniMaxH3T2VAResolution", "width", "MiniMaxH3TextToVideo", "width"),
        ("MiniMaxH3T2VAResolution", "height", "MiniMaxH3TextToVideo", "height"),
        ("MiniMaxH3TextToVideo", "positive", "BasicGuider", "conditioning"),
        ("MiniMaxH3TextToVideo", "LATENT", "SamplerCustomAdvanced", "latent_image"),
        ("MiniMaxH3TextToVideo", "actual_clip", "PreviewAny", "source"),
        ("MiniMaxH3VAEDecodeTiled", "IMAGE", "WanAutoMosaicVideo", "images"),
    ):
        link(by_type[a], out, by_type[b], inp)
    resolution.update(title="縦横比 / 元動画0.4MP（上限0.8）", widgets_values=["9:16", .4])
    by_type["RandomNoise"].update(title="Seed · 次の候補は数字変更 / 量産はrandomize", widgets_values=[42, "fixed"])
    by_type["SaveVideo"]["widgets_values"] = ["video/MiniMax_H3_07_T2VA_Master", "mp4", "h264"]
    by_type["SaveVideo"]["title"] = "元動画MP4 · 保存して06の元動画へ"
    by_type["MarkdownNote"]["widgets_values"] = [
        "## 07 · テキストだけで元動画を生成\n\n画像のアップロード不要。FULL PROMPTへ映像・音・BGMを全文で入力。"
        "<Picture 1>/<Video 1>等は使いません。初期文は『退出→空の背景→近くへ再登場』の例。\n\n"
        "初期5秒=124frames/24fps=5.167秒、9:16・約0.4MP。尺は17n+5へ切り上げ。"
        "まず短いワンカット・人物1人で試す。最大0.8MP/15秒指定、OOM回避は保証されません。\n\n"
        "DaSiWa v2を共有。RefMod・人物置換LoRA・2xはここでは使いません。"
        "INT8 VAEのネイティブ解像度で保存し、良い元動画だけ06でキャラ置換・2x仕上げ。\n\n"
        "MP4保存→06の元動画欄へアップロード。初期124framesなら06の区間は0秒/5.2秒。"
        "長い元動画は06の区間も延長。生成時と置換時の解像度を合わせる。\n\n"
        "次の候補はseedを変更。複数キューはseedのcontrol after generateをrandomizeにして投入。"
        "2回分の生成費用がかかります。時刻・動き・再登場時の同一性は保証されません。\n\n"
        "最終出力前にCPU輪郭モザイクを適用。既存04/05/06は変更なし。詳細: docs/text-to-video.md"
    ]
    layout = {
        "MiniMaxH3T2VAPrompt": (40, 80, 620, 780), "MiniMaxH3T2VAResolution": (40, 930, 620, 180),
        "MarkdownNote": (40, 1180, 620, 650), "UNETLoader": (760, 80, 540, 130),
        "MiniMaxH3SigmaShift": (760, 290, 540, 130), "CLIPLoader": (760, 500, 540, 170),
        "MiniMaxH3TextToVideo": (760, 1080, 540, 260), "PreviewAny": (760, 1420, 540, 320),
        "BasicScheduler": (1400, 80, 420, 160), "RandomNoise": (1400, 320, 420, 140),
        "KSamplerSelect": (1400, 540, 420, 90), "BasicGuider": (1400, 710, 420, 100),
        "SamplerCustomAdvanced": (1400, 890, 420, 280),
        "MiniMaxH3ReleaseVRAMLatent": (1920, 80, 500, 100),
        "MiniMaxH3VAEDecodeTiled": (1920, 260, 500, 120), "VAEDecodeAudio": (1920, 460, 500, 120),
        "WanAutoMosaicVideo": (1920, 660, 500, 480), "CreateVideo": (1920, 1220, 500, 140),
        "SaveVideo": (2520, 80, 580, 940),
    }
    for n in nodes:
        box = (760, 750 if n is video_vae else 910, 540, 100) if n["type"] == "VAELoader" else layout[n["type"]]
        n["pos"], n["size"] = list(box[:2]), list(box[2:])
        for s in n.get("inputs", []): s["link"] = None
        for s in n.get("outputs", []): s["links"] = []
    workflow["groups"] = [{"id": i+1, "title": title, "bounding": [x, 0, w, 1900],
                           "color": "#34515e", "font_size": 24, "flags": {}}
                          for i, (title, x, w) in enumerate((
                              ("01 · Text / Resolution", 10, 680), ("02 · Shared DaSiWa / T2VA", 730, 600),
                              ("03 · Sampling", 1370, 480), ("04 · Decode / CPU Mosaic", 1890, 560),
                              ("05 · 元動画を06へ", 2490, 640)))]
    by_id = {n["id"]: n for n in nodes}
    for lid, a, ai, b, bi, _ in workflow["links"]:
        by_id[a]["outputs"][ai]["links"].append(lid)
        by_id[b]["inputs"][bi]["link"] = lid
    pending, done, ordered = list(nodes), set(), []
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready: raise ValueError("T2VA graph contains a cycle")
        for n in ready:
            n["order"] = len(ordered)
            done.add(n["id"])
            ordered.append(n)
            pending.remove(n)
    workflow["nodes"] = ordered
    workflow["last_node_id"], workflow["last_link_id"] = max(by_id), max(l[0] for l in workflow["links"])
    manifest["files"] = [f for f in manifest["files"] if not f["path"].startswith("upscale_models/")]
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"])
    manifest["t2va"] = True
    return manifest, workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vae", choices=("int8", "x2-detail"), default="int8", help="Unchanged 04 VAE profile")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, workflow = build(args.profile)
    from verify_t2va import verify
    verify(workflow, manifest)
    target = args.output_dir / "minimax_h3_r2v_models.json"
    merged = merge_manifest(json.loads(target.read_text(encoding="utf-8")), manifest)
    write_json(args.output_dir / "minimax_h3_r2v_04_models.json", prepare(args.profile, vae_profile=args.vae)[0])
    write_json(target, merged)
    hf = copy.deepcopy(merged)
    hf["files"] = [f for f in hf["files"] if f.get("auth") != "civitai"]
    hf["total_bytes"] = sum(f["size"] for f in hf["files"])
    write_json(args.output_dir / "minimax_h3_r2v_hf_models.json", hf)
    write_json(args.output_dir / "minimax_h3_t2va_models.json", manifest)
    write_json(args.output_dir / "workflows" / WORKFLOW, workflow)
    print("[h3-t2va] installed 07: no references, shared DaSiWa, INT8 native-resolution decode, no extra checkpoint")


if __name__ == "__main__":
    main()
