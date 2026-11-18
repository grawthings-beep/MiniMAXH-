"""Sequential FoxHunter runner. No GPU libraries and no additional dependencies."""

import copy
import json
import re
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def build_prompt(template, image, prompt, prefix):
    """Work from the *compiled* FoxHunter graph, retaining sampler/model settings."""
    if not isinstance(template, dict) or isinstance(template.get("nodes"), list):
        raise ValueError("通常のワークフローJSONではなく Export (API) で保存したJSONを指定してください。")
    graph = copy.deepcopy(template)

    def inputs(key, kind):
        node = graph.get(key)
        if not isinstance(node, dict) or node.get("class_type") != kind:
            raise ValueError(f"{key} ({kind}) がありません。FoxHunterで Preview #2 と Final Upscale Pass をON、Single PassをOFFにしてください。")
        return node["inputs"]

    ref = inputs("5", "MiniMaxH3ReferenceToVideo")
    first = inputs("133:129", "SamplerCustomAdvanced")
    scale = inputs("604", "MiniMaxH3LatentUpscaleWithModel")
    second = inputs("135:26", "SamplerCustomAdvanced")
    save = inputs("34", "VHS_VideoCombine")
    # Prevent referencing the previous row's media or its prompt editor.
    for key in list(ref):
        if key.startswith(("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")):
            del ref[key]
    if any(key in graph for key in ("900001", "900002", "900003")):
        raise ValueError("Batch用ノードIDが既に使用されています")
    graph["900001"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    ref.update({"prompt": prompt, "ref_images.ref_image_0": ["900001", 0]})
    # Use the same video/audio VAEs as the shipped workflow even if a virtual
    # 'Anything Everywhere' wire wasn't present when exporting from the UI.
    for key, node in (("vae", "171:2"), ("audio_vae", "171:1")):
        if key not in ref:
            inputs(node, "VAELoader")
            ref[key] = [node, 0]
    conditioning = ["5", 0]
    if graph.get("459", {}).get("class_type") == "MiniMaxH3RefModApply":
        inputs("459", "MiniMaxH3RefModApply")["conditioning"] = conditioning
        conditioning = ["459", 0]
    for key in ("133:130", "135:29"):
        inputs(key, "BasicGuider")["conditioning"] = conditioning
    first.update(guider=["133:130", 0], latent_image=["5", 1])
    # These two inputs are supplied by Use Everywhere in the canvas. Some API
    # exports omit virtual wires; the normal FoxHunter video-settings subgraph
    # provides the same sampler/sigmas (the voice-only scheduler is not used).
    for key, src, kind in (("sampler", "22:6", "KSamplerSelect"), ("sigmas", "22:7", "BasicScheduler")):
        if key not in first:
            inputs(src, kind)
            first[key] = [src, 0]
    scale["samples"] = ["133:129", 1]  # #2 denoised_output, not sampler #1 or #3
    second.update(guider=["135:29", 0], latent_image=["604", 0])
    for node, kind, vae_key in (("900002", "VAEDecode", "vae"),
                                ("900003", "VAEDecodeAudio", "audio_vae")):
        graph[node] = {"class_type": kind, "inputs": {"samples": ["135:26", 0], "vae": ref[vae_key]}}
    save.update(images=["900002", 0], audio=["900003", 0], frame_rate=24,
                format="video/h264-mp4", filename_prefix=prefix, save_output=True,
                pingpong=False, loop_count=0)
    for key in ("meta_batch", "vae"):
        save.pop(key, None)

    needed = {}
    def visit(key):
        if key in needed:
            return
        entry = graph[key]
        needed[key] = entry
        for value in entry["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and isinstance(value[0], str):
                if value[0] not in graph:
                    raise ValueError(f"参照先ノード {value[0]} がありません。API形式で再出力してください。")
                visit(value[0])
    visit("34")
    if {k for k, n in needed.items() if n["class_type"] == "SamplerCustomAdvanced"} != {"133:129", "135:26"}:
        raise ValueError("サンプル2＋第2パス以外のサンプラーに依存しています")
    return needed


def call(base, path, payload=None):
    req = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"} if payload is not None else {})
    try:
        with urlopen(req, timeout=60) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read(1600).decode(errors='replace')}") from e


def await_mp4(base, job, request, pause):
    missing = 0
    while True:
        row = request(base, "/history/" + job).get(job)
        if row is not None:
            status = row.get("status", {})
            if status.get("status_str") != "success" or not status.get("completed"):
                messages = [v.get("exception_message", k) for k, v in status.get("messages", [])
                            if k in ("execution_error", "execution_interrupted")]
                raise RuntimeError("; ".join(messages) or "ComfyUIで生成失敗／中断")
            for f in row.get("outputs", {}).get("34", {}).get("gifs", []):
                if f.get("type") == "output" and f.get("filename", "").lower().endswith(".mp4"):
                    return "/".join(x for x in (f.get("subfolder"), f["filename"]) if x)
            raise RuntimeError("Final Video の保存済みMP4を確認できません")
        queue = request(base, "/queue")
        present = any(item[1] == job for name in ("queue_pending", "queue_running") for item in queue[name])
        missing = 0 if present else missing + 1
        if missing >= 3:
            raise RuntimeError("ジョブがキューと履歴から消えました。再投入せず停止します")
        pause(2)


def run_batch(template, rows, base, report, request=call, pause=time.sleep):
    # Never retry a POST: a dropped reply may already have queued a paid render.
    run_id = uuid.uuid4().hex[:12]
    for i, row in enumerate(rows):
        try:
            if i:
                queue = request(base, "/queue")
                if not queue["queue_running"] and not queue["queue_pending"]:
                    # Private Pod: ask the worker to release cached results/models.
                    # No /interrupt or queue clearing, including on failures.
                    request(base, "/free", {"free_memory": True, "unload_models": True})
            stem = re.sub(r"[^\w-]+", "_", row.get("label") or row["image"].rsplit("/", 1)[-1].rsplit(".", 1)[0])[:48]
            prefix = f"foxhunter_batch/{run_id}/{i+1:03}_{stem}"
            graph = build_prompt(template, row["image"], row["prompt"], prefix)
            report(i, "送信中", "")
            extra = {"extra_pnginfo": {"workflow": {"extra": {
                "VHS_KeepIntermediate": False, "VHS_MetadataImage": False}}}}
            job = request(base, "/prompt", {"prompt": graph, "extra_data": extra})["prompt_id"]
            report(i, "生成中", job)
            file = await_mp4(base, job, request, pause)
            report(i, "保存済み", file)
        except Exception as exc:
            report(i, "停止", str(exc))
            return
