#!/usr/bin/env python3
"""Build the opt-in, single-character R2V workflow from the pinned native template."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import uuid
from pathlib import Path

from build_workflows import add_auto_mosaic, add_upscale, write_json

ROOT = Path(__file__).resolve().parents[1]
FILENAME = "character_reveal_r2v_2x.json"
INT8_FILENAME = "character_reveal_r2v_int8_2x.json"


def with_character_refmod(workflow: dict) -> dict:
    """Upgrade only the dedicated R2VA preset. Legacy/user workflows stay valid."""
    spec = importlib.util.spec_from_file_location(
        "refmod_prompt_defaults", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/refmod_nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes = {n["id"]: n for n in workflow["nodes"]}
    edges = [(a, nodes[a]["outputs"][ai]["name"], b, nodes[b]["inputs"][bi]["name"])
             for _, a, ai, b, bi, _ in workflow["links"]]

    def replace(node_id, kind, inputs, outputs, widgets, title):
        n = nodes[node_id]
        n.update(type=kind, title=title, widgets_values=widgets,
                 properties={"Node name for S&R": kind},
                 inputs=[{"name": name, "type": type_, "link": None} for name, type_ in inputs],
                 outputs=[{"name": name, "type": type_, "links": []} for name, type_ in outputs])
        for socket in n["inputs"]:
            if socket["type"] == "INT":
                socket["widget"] = {"name": socket["name"]}

    replace(137, "MiniMaxH3RefModImages", [], [("references", "H3_CHARACTER_IMAGES")],
            ["[]", 1024, 2048], "01 · 同一キャラ1〜8枚 / 画像を追加・削除")
    replace(141, "MiniMaxH3FullPrompt", [], [("prompt", "STRING"), ("length", "INT")],
            [5.0, module.DEFAULT_FULL_PROMPT], "02 · FULL PROMPT / 全文をここ1欄へ")
    replace(142, "MiniMaxH3CreateCharacterRefMod", [("references", "H3_CHARACTER_IMAGES"), ("vae", "VAE")],
            [("character", "H3_CHARACTER_REFMOD"), ("mods", "H3_REF_MODS")], [],
            "Full RefMod · 縦横比を維持 / 追加学習なし")
    replace(136, "MiniMaxH3CharacterRefModR2V",
            [("clip", "CLIP"), ("character", "H3_CHARACTER_REFMOD"),
             ("prompt", "STRING"), ("width", "INT"), ("height", "INT"), ("length", "INT")],
            [("positive", "CONDITIONING"), ("LATENT", "LATENT"), ("reference_map", "STRING")],
            [480, 864, 124], "03 · Native R2VA + RefMod / 二重適用なし")
    save_id = max(nodes) + 1
    nodes[save_id] = {"id": save_id, "flags": {}, "order": 0, "mode": 0,
                      "pos": [40, 1070], "size": [420, 160]}
    replace(save_id, "MiniMaxH3SaveCharacterRefMod", [("mods", "H3_REF_MODS")], [("saved_path", "STRING")],
            ["character"], "RefModを書き出す / Pod削除前に保存")
    replaced_edges = []
    for a, out, b, inp in edges:
        if (a, b) == (137, 142):
            out, inp = "references", "references"
        elif (a, b) == (142, 136):
            out, inp = "character", "character"
        elif (a, b) == (119, 136):
            b, inp = 142, "vae"
        elif (a, b) == (120, 136):
            continue
        replaced_edges.append((a, out, b, inp))
    replaced_edges.append((142, "mods", save_id, "mods"))
    workflow["links"] = []
    for n in nodes.values():
        for socket in n.get("inputs", []):
            socket["link"] = None
        for socket in n.get("outputs", []):
            socket["links"] = []
    for lid, (a, out, b, inp) in enumerate(replaced_edges, 1):
        ai = next(i for i, s in enumerate(nodes[a]["outputs"]) if s["name"] == out)
        bi = next(i for i, s in enumerate(nodes[b]["inputs"]) if s["name"] == inp)
        typ = nodes[a]["outputs"][ai]["type"]
        assert typ == nodes[b]["inputs"][bi]["type"]
        nodes[a]["outputs"][ai]["links"].append(lid)
        nodes[b]["inputs"][bi]["link"] = lid
        workflow["links"].append([lid, a, ai, b, bi, typ])
    for nid, pos, size in [(137, [40, 80], [420, 670]), (142, [40, 830], [420, 170]),
                           (143, [40, 1310], [420, 440]), (136, [1920, 80], [480, 280])]:
        nodes[nid].update(pos=pos, size=size)
    nodes[143]["widgets_values"] = [
        "## 04 · Full Prompt + Character RefMod\n\n"
        "画像を1〜8枚追加。同一キャラの顔・衣装・別角度。増減・並べ替えは左のボタン。"
        "画像は開始フレームではありません。\n\n"
        "全文をFULL PROMPTへ貼るだけ。音・BGMも同じ欄。モード切替/4欄への分割は不要。"
        "<Picture 1>等の番号は左の順序。複数画像も同じキャラとして記述してください。\n\n"
        "まず5秒/0.4MP。参照の長辺1024px・合計2048tokens。枚数増加時は合計に合わせ縮小。"
        "Full Referenceでも顔保持/OOM回避は保証されません。\n\n"
        "RefModは下の保存ノードからダウンロード可能。Pod削除前に保存。"
        "ファイルは上流v5 bundle形式。追加学習/有料APIなし。"
    ]
    for group in workflow["groups"]:
        group["bounding"][3] = 1810
        if group["title"] == "02 · Direction":
            group["title"] = "02 · Full Prompt"
    pending, ordered, done = list(nodes.values()), [], set()
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("RefMod workflow contains a cycle")
        for n in ready:
            n["order"] = len(ordered)
            ordered.append(n)
            done.add(n["id"])
            pending.remove(n)
    workflow["nodes"] = ordered
    workflow["last_node_id"], workflow["last_link_id"] = max(nodes), len(workflow["links"])
    workflow["extra"]["character_r2v"].update(full_prompt=True, refmod=True, ref_total_tokens=2048)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "MiniMAXH-/character-refmod-full-v1"))
    return workflow


def add_character_decode_guard(workflow: dict) -> dict:
    """Keep the opt-in R2V's bounded decode path independent of I2V presets.

    The current main presets deliberately use native streaming/model reuse.
    Do not change their builder or reinstate its removed shared decode helper.
    """
    nodes, links = workflow["nodes"], workflow["links"]
    sampler = next(n for n in nodes if n["type"] == "SamplerCustomAdvanced")
    video = next(n for n in nodes if n["type"] == "VAEDecode")
    audio = next(n for n in nodes if n["type"] == "VAEDecodeAudio")
    decode_ids = {video["id"], audio["id"]}
    outgoing = [l for l in links if l[1] == sampler["id"] and l[3] in decode_ids and l[5] == "LATENT"]
    if len(outgoing) != 2 or {l[3] for l in outgoing} != decode_ids:
        raise ValueError("Expected sampler output to both audio and video decoders")
    guard_id = max(n["id"] for n in nodes) + 1
    guard_link = max(l[0] for l in links) + 1
    old_links = [l[0] for l in outgoing]
    for link in outgoing:
        link[1], link[2] = guard_id, 0
    output = sampler["outputs"][0]
    output["links"] = [lid for lid in output["links"] if lid not in old_links] + [guard_link]
    nodes.append({
        "id": guard_id, "type": "MiniMaxH3ReleaseVRAMLatent",
        "pos": [0, 0], "size": [500, 100], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "samples", "type": "LATENT", "link": guard_link}],
        "outputs": [{"name": "samples", "type": "LATENT", "links": old_links}],
        "title": "VRAM GUARD — unload H3 before VAE decode",
        "properties": {"Node name for S&R": "MiniMaxH3ReleaseVRAMLatent"}, "widgets_values": [],
    })
    links.append([guard_link, sampler["id"], 0, guard_id, 0, "LATENT"])
    video.update(type="MiniMaxH3VAEDecodeTiled", title="H3 VAE Decode — nested-safe native tiling", widgets_values=[])
    video.setdefault("properties", {})["Node name for S&R"] = video["type"]
    return workflow


def build(*, int8_vae: bool = False) -> dict:
    upstream = json.loads((ROOT / "workflows/upstream_minimax_h3_r2v.json").read_text(encoding="utf-8"))
    workflow = copy.deepcopy(upstream)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "MiniMAXH-/character-reveal-r2v-v1"))
    workflow.pop("definitions", None)
    remove_ids = {116, 117, 131, 132, 138, 139, 140}
    workflow["nodes"] = [n for n in workflow["nodes"] if n["id"] not in remove_ids]
    workflow["links"] = []
    # Existing tail builders need an output group; final layout replaces it below.
    workflow["groups"] = [{"title": "Output", "bounding": [-5000, 0, 15000, 10000]}]
    for node in workflow["nodes"]:
        for socket in node.get("inputs", []):
            socket["link"] = None
        for socket in node.get("outputs", []):
            socket["links"] = []
    by_id = {n["id"]: n for n in workflow["nodes"]}

    spec = importlib.util.spec_from_file_location(
        "character_prompt_defaults", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/character_nodes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def node(node_id, kind, inputs, outputs, values, title):
        item = {
            "id": node_id, "type": kind, "pos": [0, 0], "size": [400, 200],
            "flags": {}, "order": 0, "mode": 0, "title": title,
            "inputs": [{"name": name, "type": type_, "link": None} for name, type_ in inputs],
            "outputs": [{"name": name, "type": type_, "links": []} for name, type_ in outputs],
            "properties": {"Node name for S&R": kind}, "widgets_values": values,
        }
        workflow["nodes"].append(item)
        by_id[node_id] = item
        return item

    node(141, "MiniMaxH3CharacterPrompt", [], [("prompt", "STRING"), ("length", "INT")],
         [5.0, "direction", module.IDENTITY_NOTES, module.DEFAULT_DIRECTION,
          "Quiet room ambience, soft footsteps and a door opening. No dialogue.", "N/A"],
         "02 · 演出を入力 / Direction + duration (no API)")
    node(142, "MiniMaxH3CharacterReference", [("image", "IMAGE")], [("IMAGE", "IMAGE")],
         [1024], "Reference budget · 1024px long edge / OOM時は768")
    node(143, "MarkdownNote", [], [], [
        "## 04 Character Reveal · R2V Quality 2x\n\n"
        "キャラ画像は外見の参照。冒頭フレームには固定しません。\n\n"
        "1. 左上へ1人のキャラ画像をアップロード。\n"
        "2. 演出ノードの direction に、冒頭→途中→見せ場を具体的に入力。"
        "英語推奨。自動翻訳・LLM・有料APIはありません。\n"
        "3. まず5秒 / 0.4MP / 25 stepsで動作と顔を確認。"
        "構成確認後に7秒、0.6MP、最終0.98MPへ一つずつ変更。OOMしない保証はありません。\n\n"
        "`direction` は公式形式へ整形。`full_prompt` は direction の全文をそのまま送信。"
        "後者では identity_notes/soundscape/music は無視されます。\n\n"
        "キャラ保持と時刻指定は強制ではありません。1枚の参照で大きな後ろ向き回転を"
        "求めない。小さな顔しかない全身絵より、顔と衣装が読める画像から試す。\n\n"
        "専用ref2vaモデル / res_multistep + normal / LoRA・近似キャッシュなし。"
        "既存のFL2VA Turbo LoRAはこのグラフへ流用しない。\n\n"
        "参照は最大1024pxに制限して native `max` へ渡すため、無制限な2K参照ではありません。"
        "元画像の拡大・切り抜きはしません（32px格子への丸めあり）。"
    ], "使い方 / Read before generation")
    by_id[137]["widgets_values"] = ["", "image"]
    by_id[137]["title"] = "01 · キャラ画像 / Appearance reference, NOT first frame"
    by_id[115]["widgets_values"] = ["9:16 (Portrait Widescreen)", 0.4, 32]
    by_id[115]["title"] = "Output resolution · 初回0.4 / 最終候補0.6–0.98MP"
    by_id[124]["widgets_values"] = ["normal", 25, 1.0]
    by_id[124]["title"] = "Quality sampling · 25 steps (20 for comparison)"
    by_id[129]["widgets_values"] = [42, "fixed"]
    by_id[129]["title"] = "Seed · fixed for comparisons / 次の候補は変更"
    by_id[136]["widgets_values"] = ["", 480, 864, 124, "max"]
    by_id[136]["title"] = "03 · Native R2V · キャラ参照（開始フレームではない）"
    by_id[92]["widgets_values"] = ["video/MiniMax_H3_04_Character_R2V_2x", "mp4", "h264"]

    def connect(origin, output, target, input_):
        a, b = by_id[origin], by_id[target]
        ai = next(i for i, s in enumerate(a["outputs"]) if s["name"] == output)
        bi = next(i for i, s in enumerate(b["inputs"]) if s["name"] == input_)
        lid = len(workflow["links"]) + 1
        workflow["links"].append([lid, origin, ai, target, bi, a["outputs"][ai]["type"]])
        a["outputs"][ai]["links"].append(lid)
        b["inputs"][bi]["link"] = lid

    for edge in [
        (137, "IMAGE", 142, "image"), (142, "IMAGE", 136, "ref_images.ref_image_0"),
        (141, "prompt", 136, "prompt"), (141, "length", 136, "length"),
        (115, "width", 136, "width"), (115, "height", 136, "height"),
        (128, "CLIP", 136, "clip"), (119, "VAE", 136, "vae"), (120, "VAE", 136, "audio_vae"),
        (127, "MODEL", 124, "model"), (127, "MODEL", 126, "model"),
        (136, "positive", 126, "conditioning"), (136, "LATENT", 125, "latent_image"),
        (129, "NOISE", 125, "noise"), (126, "GUIDER", 125, "guider"),
        (123, "SAMPLER", 125, "sampler"), (124, "SIGMAS", 125, "sigmas"),
        (125, "output", 122, "samples"), (125, "output", 121, "samples"),
        (119, "VAE", 122, "vae"), (120, "VAE", 121, "vae"),
        (122, "IMAGE", 130, "images"), (121, "AUDIO", 130, "audio"),
        (130, "VIDEO", 92, "video"),
    ]:
        connect(*edge)
    workflow = add_character_decode_guard(add_auto_mosaic(add_upscale(workflow, "Character R2V"), "Character R2V"))
    # Every editable node is top-level; deterministic, disjoint left-to-right columns.
    layouts = {
        137: (40, 80, 420, 370), 142: (40, 510, 420, 100), 143: (40, 680, 420, 880),
        141: (560, 80, 620, 1100), 115: (560, 1250, 620, 180),
        127: (1280, 80, 540, 120), 128: (1280, 270, 540, 150),
        119: (1280, 490, 540, 100), 120: (1280, 660, 540, 100),
        136: (1920, 80, 480, 680),
        124: (2500, 80, 420, 150), 129: (2500, 300, 420, 110),
        123: (2500, 480, 420, 80), 126: (2500, 630, 420, 100), 125: (2500, 800, 420, 240),
        122: (3020, 250, 500, 100), 121: (3020, 420, 500, 100),
        130: (3020, 1270, 500, 110), 92: (3620, 80, 540, 650),
    }
    additional = {
        "UpscaleModelLoader": (1280, 830, 540, 100),
        "MiniMaxH3ReleaseVRAMLatent": (3020, 80, 500, 100),
        "ImageUpscaleWithModel": (3020, 590, 500, 100),
        "WanAutoMosaicVideo": (3020, 760, 500, 440),
    }
    for n in workflow["nodes"]:
        x, y, w, h = layouts[n["id"]] if n["id"] in layouts else additional[n["type"]]
        n["pos"], n["size"] = [x, y], [w, h]
    workflow["groups"] = [
        {"id": i+1, "title": title, "bounding": [x, 0, w, 1630], "color": "#34515e", "font_size": 24, "flags": {}}
        for i, (title, x, w) in enumerate([
            ("01 · Character", 10, 480), ("02 · Direction", 530, 680),
            ("Models · Ref2VA", 1250, 600), ("03 · Reference conditioning", 1890, 540),
            ("04 · Quality sampling", 2470, 480), ("05 · Tiled decode / 2x / mosaic", 2990, 560),
            ("06 · MP4", 3590, 600),
        ])
    ]
    # Topological order (rather than canvas coordinates) for stable UI serialization.
    pending = list(workflow["nodes"])
    done = set()
    ordered = []
    while pending:
        ready = [n for n in pending if all(l[1] in done for l in workflow["links"] if l[3] == n["id"])]
        if not ready:
            raise ValueError("Workflow contains a cycle")
        for n in ready:
            n["order"] = len(ordered)
            ordered.append(n)
            done.add(n["id"])
            pending.remove(n)
    workflow["nodes"] = ordered
    workflow["last_node_id"] = max(n["id"] for n in ordered)
    workflow["last_link_id"] = max(l[0] for l in workflow["links"])
    workflow["extra"] = {"ds": {"scale": 0.65, "offset": [30, 30]}, "character_r2v": {
        "appearance_reference_not_first_frame": True, "quality_tested_on_gpu": False,
        "no_external_prompt_api": True, "ref_long_edge_cap": 1024,
    }}
    if int8_vae:
        manifest = json.loads((ROOT / "manifests/minimax_h3_r2v_int8_upscale.json").read_text(encoding="utf-8"))
        workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "MiniMAXH-/character-r2v-int8-v1"))
        video_vae = next(n for n in ordered if n["id"] == 119)
        name = "minimax_h3_video_vae_int8_convrot.safetensors"
        video_vae["widgets_values"] = [name]
        video_vae["title"] = "Video VAE · INT8 ConvRot / native tiled decode"
        video_vae["properties"]["models"] = [{
            "name": name, "directory": "vae",
            "url": f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/vae/{name}",
        }]
        workflow["extra"]["character_r2v"]["runtime_profile"] = "r2v"
        workflow["extra"]["character_r2v"]["video_vae"] = name
        note = next(n for n in ordered if n["id"] == 143)
        note["widgets_values"][0] += (
            "\n\nR2VA専用起動: H3_PROFILE=r2v。必要な5モデルとCPUモザイクだけを取得。"
            "INT8 VAE使用。生成本体・キャラ参照・演出入力は従来と同じです。"
            "別の構図画像、追加LoRA、Turboは不要です。"
        )
    return with_character_refmod(workflow) if int8_vae else workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workflows")
    parser.add_argument("--int8-vae", action="store_true")
    args = parser.parse_args()
    write_json(args.output_dir / (INT8_FILENAME if args.int8_vae else FILENAME), build(int8_vae=args.int8_vae))
