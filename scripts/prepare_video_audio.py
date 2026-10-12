#!/usr/bin/env python3
"""Opt-in 08: H3 audio-only sampling from an uploaded, visually pinned MP4."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import uuid

from prepare_r2v_profile import ROOT, prepare, write_json
from prepare_upscale_compare import merge_manifest

WORKFLOW = "08_MiniMax_H3_Video_to_Audio.json"


def build(profile: str, project: Path = ROOT):
    if profile not in {"official", "dasiwa-v2"}:
        raise ValueError("H3_VIDEO_AUDIO supports official or dasiwa-v2, not Turbo.")
    manifest, _, baseline = prepare(profile, project, vae_profile="int8")
    manifest["files"] = [f for f in manifest["files"] if not f["path"].startswith("upscale_models/")]
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"])
    manifest["video_audio"] = True
    keep = {"UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3SigmaShift", "BasicScheduler",
            "KSamplerSelect", "RandomNoise", "BasicGuider", "SamplerCustomAdvanced",
            "MiniMaxH3ReleaseVRAMLatent", "VAEDecodeAudio", "MarkdownNote"}
    nodes = [copy.deepcopy(n) for n in baseline["nodes"] if n["type"] in keep]
    by_type = {n["type"]: n for n in nodes}
    video_vae = next(n for n in nodes if n["type"] == "VAELoader" and "video_vae" in n["widgets_values"][0])
    audio_vae = next(n for n in nodes if n["type"] == "VAELoader" and "audio_vae" in n["widgets_values"][0])
    spec = importlib.util.spec_from_file_location(
        "video_audio_defaults", project / "custom_nodes/minimax_h3_ordered_storyboard/video_audio_nodes.py")
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)

    def add(kind, inputs=(), outputs=(), widgets=(), title=None):
        item = {"id": max(n["id"] for n in nodes) + 1, "type": kind, "title": title or kind,
                "pos": [0, 0], "size": [440, 140], "flags": {}, "order": 0, "mode": 0,
                "properties": {"Node name for S&R": kind}, "widgets_values": list(widgets),
                "inputs": [{"name": name, "type": typ, "link": None} for name, typ in inputs],
                "outputs": [{"name": name, "type": typ, "links": []} for name, typ in outputs]}
        nodes.append(item)
        by_type[kind] = item
        return item

    add("LoadVideo", outputs=(("VIDEO", "VIDEO"),), widgets=("",), title="01 · 完成MP4をアップロード")
    add("MiniMaxH3VideoAudioClip", inputs=(("video", "VIDEO"),),
        outputs=(("source", "H3_AUDIO_CLIP"), ("actual_clip", "STRING")), widgets=(.2,),
        title="解析用0.2MP / 元MP4は変更しない")
    add("PreviewAny", inputs=(("source", "*"),), outputs=(("STRING", "STRING"),),
        title="実際の解析尺・解像度")
    add("MiniMaxH3VideoAudioPrompt", outputs=(("STRING", "STRING"),),
        widgets=(defaults.DEFAULT_AUDIO_PROMPT,), title="02 · 効果音・吐息の指示（全文1欄）")
    add("MiniMaxH3VideoAudioCondition",
        inputs=(("clip", "CLIP"), ("vae", "VAE"), ("source", "H3_AUDIO_CLIP"), ("prompt", "STRING")),
        outputs=(("positive", "CONDITIONING"), ("LATENT", "LATENT")),
        title="03 · 映像latent固定 / 音声だけ生成")
    add("MiniMaxH3VideoAudioSave", inputs=(("source", "H3_AUDIO_CLIP"), ("generated", "AUDIO")),
        outputs=(("video", "VIDEO"), ("saved_path", "STRING")),
        widgets=("video/MiniMax_H3_08_Video_Audio",), title="04 · 元映像をコピー + H3音声で保存")
    by_type["MarkdownNote"]["widgets_values"] = [
        "## 08 · 完成動画にH3で効果音・吐息を追加\n\n"
        "4.5〜15秒のMP4を左へアップロード。MP3や声の見本は不要。音の指示を全文1欄へ。"
        "映像は解析時だけ24fps・初期0.2MPへ縮小し、H3の映像latentは固定。音声だけ生成します。\n\n"
        "出力時は元MP4の映像ストリームをコピーし、生成音声へ差し替えます。"
        "元の音声を残すミックスではありません。映像を再エンコードしません。\n\n"
        "音のタイミング・吐息の自然さ・不要な声の抑制は保証できません。"
        "実動画で確認し、合わなければseedや音の指示を変えてください。"
        "長尺は動画を15秒以下に分割。映像VAEの一度のencodeとH3推論が必要で、即時処理ではありません。\n\n"
        "DaSiWa/公式Ref2VAとQwen、標準INT8映像VAE、音声VAEを共有。"
        "X2 Detail設定でもこの08だけ標準INT8 VAEを追加取得します。"
        "04/05/06/07は変更しません。docs/video-to-audio.md"
    ]
    by_type["BasicScheduler"]["title"] = "H3 audio sampling · pinned video mask"
    by_type["RandomNoise"]["title"] = "Seed · 音が合わなければ変更"
    by_type["VAEDecodeAudio"]["title"] = "H3生成音声だけデコード"
    by_type["MiniMaxH3ReleaseVRAMLatent"]["title"] = "H3を解放 → 音声デコード"

    workflow = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"MiniMAXH-/video-to-audio/{profile}/v1")),
                "revision": 0, "version": 0.4, "nodes": nodes, "links": [], "groups": [], "config": {},
                "extra": {"ds": {"scale": .7, "offset": [30, 30]},
                          "video_audio": {"model_profile": profile, "video_noise_mask": 0,
                                          "audio_noise_mask": 1, "original_video_stream_copy": True,
                                          "quality_tested_on_gpu": False}}}

    def link(a, out, b, inp):
        origin, target = by_type[a], by_type[b]
        ai = next(i for i, socket in enumerate(origin["outputs"]) if socket["name"] == out)
        bi = next(i for i, socket in enumerate(target["inputs"]) if socket["name"] == inp)
        typ = origin["outputs"][ai]["type"]
        assert target["inputs"][bi]["type"] in (typ, "*")
        workflow["links"].append([len(workflow["links"]) + 1, origin["id"], ai, target["id"], bi, typ])

    for edge in (
        ("LoadVideo", "VIDEO", "MiniMaxH3VideoAudioClip", "video"),
        ("MiniMaxH3VideoAudioClip", "actual_clip", "PreviewAny", "source"),
        ("CLIPLoader", "CLIP", "MiniMaxH3VideoAudioCondition", "clip"),
        ("MiniMaxH3VideoAudioPrompt", "STRING", "MiniMaxH3VideoAudioCondition", "prompt"),
        ("MiniMaxH3VideoAudioClip", "source", "MiniMaxH3VideoAudioCondition", "source"),
        ("MiniMaxH3VideoAudioCondition", "positive", "BasicGuider", "conditioning"),
        ("MiniMaxH3VideoAudioCondition", "LATENT", "SamplerCustomAdvanced", "latent_image"),
        ("BasicScheduler", "SIGMAS", "SamplerCustomAdvanced", "sigmas"),
        ("KSamplerSelect", "SAMPLER", "SamplerCustomAdvanced", "sampler"),
        ("RandomNoise", "NOISE", "SamplerCustomAdvanced", "noise"),
        ("BasicGuider", "GUIDER", "SamplerCustomAdvanced", "guider"),
        ("SamplerCustomAdvanced", "output", "MiniMaxH3ReleaseVRAMLatent", "samples"),
        ("MiniMaxH3ReleaseVRAMLatent", "samples", "VAEDecodeAudio", "samples"),
        ("VAEDecodeAudio", "AUDIO", "MiniMaxH3VideoAudioSave", "generated"),
        ("MiniMaxH3VideoAudioClip", "source", "MiniMaxH3VideoAudioSave", "source"),
    ):
        link(*edge)
    if "MiniMaxH3SigmaShift" in by_type:
        link("UNETLoader", "MODEL", "MiniMaxH3SigmaShift", "model")
        model_source = "MiniMaxH3SigmaShift"
    else:
        model_source = "UNETLoader"
    link(model_source, "MODEL", "BasicScheduler", "model")
    link(model_source, "MODEL", "BasicGuider", "model")
    # VAE loader IDs must be routed directly: there are two nodes of this type.
    for origin, target, name in ((video_vae, by_type["MiniMaxH3VideoAudioCondition"], "vae"),
                                 (audio_vae, by_type["VAEDecodeAudio"], "vae")):
        bi = next(i for i, socket in enumerate(target["inputs"]) if socket["name"] == name)
        workflow["links"].append([len(workflow["links"]) + 1, origin["id"], 0, target["id"], bi, "VAE"])

    layout = {"LoadVideo": (40, 80, 560, 650), "MiniMaxH3VideoAudioClip": (40, 790, 560, 220),
              "PreviewAny": (40, 1080, 560, 230), "MarkdownNote": (40, 1380, 560, 600),
              "MiniMaxH3VideoAudioPrompt": (700, 80, 620, 850),
              "UNETLoader": (1420, 80, 540, 140), "CLIPLoader": (1420, 300, 540, 160),
              "MiniMaxH3SigmaShift": (1420, 540, 540, 140),
              "MiniMaxH3VideoAudioCondition": (2060, 80, 540, 480),
              "BasicScheduler": (2700, 80, 460, 170), "KSamplerSelect": (2700, 330, 460, 90),
              "RandomNoise": (2700, 500, 460, 130), "BasicGuider": (2700, 710, 460, 100),
              "SamplerCustomAdvanced": (2700, 890, 460, 260),
              "MiniMaxH3ReleaseVRAMLatent": (3260, 80, 540, 110),
              "VAEDecodeAudio": (3260, 270, 540, 120),
              "MiniMaxH3VideoAudioSave": (3900, 80, 600, 400)}
    for node in nodes:
        box = (1420, 760 if node is video_vae else 930, 540, 100) if node["type"] == "VAELoader" else layout[node["type"]]
        node["pos"], node["size"] = list(box[:2]), list(box[2:])
        for socket in node.get("inputs", []): socket["link"] = None
        for socket in node.get("outputs", []): socket["links"] = []
    workflow["groups"] = [
        {"id": i+1, "title": title, "bounding": [x, 0, width, 2050],
         "color": "#34515e", "font_size": 24, "flags": {}}
        for i, (title, x, width) in enumerate((
            ("01 · 元動画", 10, 620), ("02 · 音の指示", 670, 680),
            ("03 · 共有モデル / 解析", 1390, 600), ("04 · 映像固定", 2030, 600),
            ("05 · 音声生成", 2670, 520), ("06 · 音声デコード", 3230, 600),
            ("07 · 元映像のままMP4", 3870, 660)))]
    by_id = {n["id"]: n for n in nodes}
    for lid, a, ai, b, bi, _ in workflow["links"]:
        by_id[a]["outputs"][ai]["links"].append(lid)
        by_id[b]["inputs"][bi]["link"] = lid
    pending, done, ordered = list(nodes), set(), []
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready: raise ValueError("Video Audio workflow contains a cycle")
        for node in ready:
            node["order"] = len(ordered)
            ordered.append(node)
            done.add(node["id"])
            pending.remove(node)
    workflow["nodes"] = ordered
    workflow["last_node_id"] = max(by_id)
    workflow["last_link_id"] = len(workflow["links"])
    return manifest, workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vae", choices=("int8", "x2-detail"), required=True,
                        help="04's VAE profile; 08 always uses a separate standard INT8 VAE")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, workflow = build(args.profile)
    from verify_video_audio import verify
    verify(workflow, manifest)
    destination = args.output_dir
    current = json.loads((destination / "minimax_h3_r2v_models.json").read_text(encoding="utf-8"))
    merged = merge_manifest(current, manifest)
    write_json(destination / "minimax_h3_r2v_04_models.json", prepare(args.profile, vae_profile=args.vae)[0])
    write_json(destination / "minimax_h3_r2v_models.json", merged)
    hf = copy.deepcopy(merged)
    hf["files"] = [f for f in hf["files"] if f.get("auth") != "civitai"]
    hf["total_bytes"] = sum(f["size"] for f in hf["files"])
    write_json(destination / "minimax_h3_r2v_hf_models.json", hf)
    write_json(destination / "minimax_h3_video_audio_models.json", manifest)
    write_json(destination / "workflows" / WORKFLOW, workflow)
    print(f"[h3-video-audio] installed 08; {len(merged['files'])} unique model assets")


if __name__ == "__main__":
    main()
