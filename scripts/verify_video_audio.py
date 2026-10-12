"""Static contract for the opt-in original-video + generated-audio workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare
from verify_workflow import workflow_models, verify_comfyui_nodes


def verify(workflow, manifest, comfyui_root=None):
    nodes, raw_links = workflow["nodes"], workflow["links"]
    by_id, links = {n["id"]: n for n in nodes}, {l[0]: l for l in raw_links}
    if len(by_id) != len(nodes) or len(links) != len(raw_links):
        raise RuntimeError("Duplicate video-audio node/link ID")
    allowed = {"LoadVideo", "MiniMaxH3VideoAudioClip", "PreviewAny", "MarkdownNote",
               "MiniMaxH3VideoAudioPrompt", "UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3R2VLoRA",
               "MiniMaxH3SigmaShift", "MiniMaxH3VideoAudioCondition", "BasicScheduler",
               "KSamplerSelect", "RandomNoise", "BasicGuider", "SamplerCustomAdvanced",
               "MiniMaxH3ReleaseVRAMLatent", "VAEDecodeAudio", "MiniMaxH3VideoAudioSave"}
    profile = manifest.get("r2v_profile") or "official"
    expected_types = allowed if profile == "dasiwa-v2" else allowed - {"MiniMaxH3SigmaShift"}
    if {n["type"] for n in nodes} != expected_types or workflow.get("definitions"):
        raise RuntimeError("08 must contain only audio generation, one optional LoRA and no video processing/subgraphs")
    for n in nodes:
        if n.get("mode", 0) != 0:
            raise RuntimeError("Disabled node in 08")
        for index, socket in enumerate(n.get("inputs", [])):
            lid = socket.get("link")
            if lid is not None and (lid not in links or links[lid][3:5] != [n["id"], index]):
                raise RuntimeError("Serialized input link mismatch")
        for index, socket in enumerate(n.get("outputs", [])):
            for lid in socket.get("links") or []:
                if lid not in links or links[lid][1:3] != [n["id"], index]:
                    raise RuntimeError("Serialized output link mismatch")
    for lid, a, ai, b, bi, typ in links.values():
        if a not in by_id or b not in by_id:
            raise RuntimeError("Dangling 08 link")
        out, inp = by_id[a]["outputs"][ai], by_id[b]["inputs"][bi]
        if out["type"] != typ or inp["type"] not in (typ, "*") or inp["link"] != lid or out["links"].count(lid) != 1:
            raise RuntimeError("Broken 08 link")
        if by_id[a]["order"] >= by_id[b]["order"]:
            raise RuntimeError("Non-topological 08 workflow")

    def one(kind):
        found = [n for n in nodes if n["type"] == kind]
        if len(found) != 1:
            raise RuntimeError(f"Expected exactly one {kind}")
        return found[0]

    def route(target, name, origin, slot=0):
        input_ = next(i for i in target["inputs"] if i["name"] == name)
        if input_["link"] not in links or links[input_["link"]][1:3] != [origin["id"], slot]:
            raise RuntimeError(f"Incorrect route: {target['type']}.{name}")

    if profile not in {"official", "dasiwa-v2"} or manifest.get("video_audio") is not True:
        raise RuntimeError("08 needs a supported Ref2VA model")
    reference, _, baseline = prepare(profile, vae_profile="int8")
    expected = [f for f in reference["files"] if not f["path"].startswith("upscale_models/")]
    if manifest["files"] != expected or manifest["total_bytes"] != sum(f["size"] for f in expected):
        raise RuntimeError("08 model manifest is not the shared four-model set")
    if workflow_models(nodes) != {f["path"] for f in expected}:
        raise RuntimeError("08 workflow model dependencies differ from manifest")
    baseline_nodes = {n["type"]: n for n in baseline["nodes"]}
    for kind in ("UNETLoader", "CLIPLoader", "BasicScheduler", "KSamplerSelect"):
        if one(kind)["widgets_values"] != baseline_nodes[kind]["widgets_values"]:
            raise RuntimeError(f"Changed shared model/sampling settings: {kind}")
    lora = one("MiniMaxH3R2VLoRA")
    if lora["widgets_values"] != baseline_nodes["MiniMaxH3R2VLoRA"]["widgets_values"] or lora["widgets_values"][2] is not False:
        raise RuntimeError("08 LoRA must default to the same OFF state as 04")
    if profile == "dasiwa-v2" and one("MiniMaxH3SigmaShift")["widgets_values"] != baseline_nodes["MiniMaxH3SigmaShift"]["widgets_values"]:
        raise RuntimeError("Changed DaSiWa AV shift")
    vaes = {n["widgets_values"][0]: n for n in nodes if n["type"] == "VAELoader"}
    if set(vaes) != {"minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"}:
        raise RuntimeError("08 requires the standard encode VAE and H3 audio VAE")
    source, cond, prompt = one("MiniMaxH3VideoAudioClip"), one("MiniMaxH3VideoAudioCondition"), one("MiniMaxH3VideoAudioPrompt")
    route(source, "video", one("LoadVideo"))
    route(one("PreviewAny"), "source", source, 1)
    route(cond, "source", source)
    route(cond, "prompt", prompt)
    route(cond, "clip", one("CLIPLoader"))
    route(cond, "vae", vaes["minimax_h3_video_vae_int8_convrot.safetensors"])
    model_source = one("UNETLoader")
    route(lora, "model", model_source)
    model_source = lora
    if profile == "dasiwa-v2":
        shift = one("MiniMaxH3SigmaShift")
        route(shift, "model", model_source)
        model_source = shift
    route(one("BasicScheduler"), "model", model_source)
    guide = one("BasicGuider")
    route(guide, "model", model_source)
    route(guide, "conditioning", cond)
    sampler, guard, decoder, saver = (one(k) for k in ("SamplerCustomAdvanced", "MiniMaxH3ReleaseVRAMLatent",
                                                    "VAEDecodeAudio", "MiniMaxH3VideoAudioSave"))
    for name, origin, slot in (("latent_image", cond, 1), ("guider", guide, 0),
                               ("sigmas", one("BasicScheduler"), 0), ("sampler", one("KSamplerSelect"), 0),
                               ("noise", one("RandomNoise"), 0)):
        route(sampler, name, origin, slot)
    route(guard, "samples", sampler)
    route(decoder, "samples", guard)
    route(decoder, "vae", vaes["minimax_h3_audio_vae_fp32.safetensors"])
    route(saver, "source", source)
    route(saver, "generated", decoder)
    if workflow.get("extra", {}).get("video_audio", {}).get("original_video_stream_copy") is not True:
        raise RuntimeError("08 must preserve original compressed video")
    groups = [g["bounding"] for g in workflow["groups"]]
    boxes = [n["pos"] + n["size"] for n in nodes]
    for collection in (groups, boxes):
        for i, (x, y, w, h) in enumerate(collection):
            for u, v, p, q in collection[i+1:]:
                if max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q):
                    raise RuntimeError("08 canvas overlap")
    for x, y, w, h in boxes:
        if sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh for gx, gy, gw, gh in groups) != 1:
            raise RuntimeError("08 node outside its group")
    if comfyui_root:
        verify_comfyui_nodes(comfyui_root, expected_types, set(), [ROOT / "custom_nodes/minimax_h3_ordered_storyboard"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.workflow.read_text(encoding="utf-8")),
           json.loads(args.manifest.read_text(encoding="utf-8")), args.comfyui_root)
    print("Video-to-audio workflow / model dependencies / pinned-video route verified")
