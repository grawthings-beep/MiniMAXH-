"""Verify I2VA first-frame routing, shared model assets, and finished-frame processing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare
from verify_workflow import verify_comfyui_nodes, workflow_models


def verify(workflow: dict, manifest: dict, comfyui_root: Path | None = None) -> None:
    nodes = workflow["nodes"]
    by_id = {n["id"]: n for n in nodes}
    links = {l[0]: l for l in workflow["links"]}
    if len(by_id) != len(nodes) or len(links) != len(workflow["links"]) or workflow.get("definitions"):
        raise RuntimeError("I2VA graph has duplicate IDs or an unexpected subgraph")
    for lid, origin, out_slot, target, in_slot, kind in links.values():
        if origin not in by_id or target not in by_id:
            raise RuntimeError("I2VA dangling link")
        source, dest = by_id[origin], by_id[target]
        output, input_ = source["outputs"][out_slot], dest["inputs"][in_slot]
        if output["type"] != kind or input_["type"] not in {kind, "*"}:
            raise RuntimeError(f"I2VA link {lid} type mismatch")
        if input_["link"] != lid or output["links"].count(lid) != 1:
            raise RuntimeError(f"I2VA link {lid} serialized socket mismatch")
        if source["order"] >= dest["order"]:
            raise RuntimeError("I2VA order is not topological")
    for node in nodes:
        if node.get("mode") != 0:
            raise RuntimeError("I2VA nodes must all be active")
        for slot, input_ in enumerate(node.get("inputs", [])):
            lid = input_.get("link")
            if lid is not None and (lid not in links or links[lid][3:5] != [node["id"], slot]):
                raise RuntimeError("I2VA dangling input socket")
        for slot, output in enumerate(node.get("outputs", [])):
            for lid in output.get("links") or []:
                if lid not in links or links[lid][1:3] != [node["id"], slot]:
                    raise RuntimeError("I2VA dangling output socket")

    def one(kind: str) -> dict:
        selected = [n for n in nodes if n["type"] == kind]
        if len(selected) != 1:
            raise RuntimeError(f"I2VA expected exactly one {kind}")
        return selected[0]

    def route(target: dict, socket: str, source: dict, out_slot: int = 0) -> None:
        input_ = next(i for i in target["inputs"] if i["name"] == socket)
        if input_["link"] not in links or links[input_["link"]][1:3] != [source["id"], out_slot]:
            raise RuntimeError(f"I2VA wrong route to {target['type']}.{socket}")

    expected_types = {"LoadImage", "MiniMaxH3I2VAPrompt", "MiniMaxH3T2VAResolution",
                      "MiniMaxH3FirstFrameToVideo", "UNETLoader", "CLIPLoader", "VAELoader",
                      "MiniMaxH3SigmaShift", "BasicScheduler", "KSamplerSelect", "RandomNoise",
                      "BasicGuider", "SamplerCustomAdvanced", "MiniMaxH3ReleaseVRAMLatent",
                      "MiniMaxH3VAEDecodeTiled", "VAEDecodeAudio", "UpscaleModelLoader",
                      "ImageUpscaleWithModel", "WanAutoMosaicVideo", "CreateVideo", "SaveVideo",
                      "PreviewAny", "MarkdownNote"}
    if {n["type"] for n in nodes} != expected_types:
        raise RuntimeError("I2VA contains unexpected or missing node classes")
    if manifest.get("r2v_profile") != "dasiwa-v2" or not manifest.get("i2va"):
        raise RuntimeError("I2VA must share the DaSiWa v2 hybrid profile")
    baseline_manifest, _, baseline = prepare("dasiwa-v2", vae_profile="int8")
    if manifest["files"] != baseline_manifest["files"] or manifest["total_bytes"] != baseline_manifest["total_bytes"]:
        raise RuntimeError("I2VA assets must be exactly the pinned INT8 R2VA/upscale asset set")
    if workflow_models(nodes) != {f["path"] for f in manifest["files"]} | {
        "auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"}:
        raise RuntimeError("I2VA model names do not match the manifest")
    baseline_types = {n["type"]: n for n in baseline["nodes"]}
    for kind in ("UNETLoader", "CLIPLoader", "MiniMaxH3SigmaShift", "BasicScheduler",
                 "KSamplerSelect", "WanAutoMosaicVideo", "UpscaleModelLoader"):
        if one(kind)["widgets_values"] != baseline_types[kind]["widgets_values"]:
            raise RuntimeError(f"I2VA changed pinned {kind} settings")

    cond, image, prompt = one("MiniMaxH3FirstFrameToVideo"), one("LoadImage"), one("MiniMaxH3I2VAPrompt")
    video_vae = next(n for n in nodes if n["type"] == "VAELoader" and "video_vae" in n["widgets_values"][0])
    audio_vae = next(n for n in nodes if n["type"] == "VAELoader" and "audio_vae" in n["widgets_values"][0])
    if len([n for n in nodes if n["type"] == "VAELoader"]) != 2 or image["widgets_values"][0] != "":
        raise RuntimeError("I2VA needs an empty first-frame upload and exactly two VAEs")
    route(cond, "first_frame", image)
    route(cond, "vae", video_vae)
    route(cond, "clip", one("CLIPLoader"))
    route(cond, "prompt", prompt)
    route(cond, "length", prompt, 1)
    for slot, name in enumerate(("width", "height")):
        route(cond, name, one("MiniMaxH3T2VAResolution"), slot)
    route(one("PreviewAny"), "source", cond, 2)
    shift = one("MiniMaxH3SigmaShift")
    route(shift, "model", one("UNETLoader"))
    route(one("BasicScheduler"), "model", shift)
    route(one("BasicGuider"), "model", shift)
    route(one("BasicGuider"), "conditioning", cond)
    sampler, guard = one("SamplerCustomAdvanced"), one("MiniMaxH3ReleaseVRAMLatent")
    for name, source, slot in (("latent_image", cond, 1), ("guider", one("BasicGuider"), 0),
                               ("sigmas", one("BasicScheduler"), 0), ("sampler", one("KSamplerSelect"), 0),
                               ("noise", one("RandomNoise"), 0)):
        route(sampler, name, source, slot)
    route(guard, "samples", sampler)
    route(one("MiniMaxH3VAEDecodeTiled"), "samples", guard)
    route(one("MiniMaxH3VAEDecodeTiled"), "vae", video_vae)
    route(one("VAEDecodeAudio"), "samples", guard)
    route(one("VAEDecodeAudio"), "vae", audio_vae)
    route(one("ImageUpscaleWithModel"), "upscale_model", one("UpscaleModelLoader"))
    route(one("ImageUpscaleWithModel"), "image", one("MiniMaxH3VAEDecodeTiled"))
    route(one("WanAutoMosaicVideo"), "images", one("ImageUpscaleWithModel"))
    route(one("CreateVideo"), "images", one("WanAutoMosaicVideo"))
    route(one("CreateVideo"), "audio", one("VAEDecodeAudio"))
    route(one("SaveVideo"), "video", one("CreateVideo"))
    if one("CreateVideo")["widgets_values"][0] != 24:
        raise RuntimeError("I2VA output must use 24fps")

    groups = [g["bounding"] for g in workflow["groups"]]
    boxes = [n["pos"] + n["size"] for n in nodes]
    for collection in (groups, boxes):
        for i, (x, y, w, h) in enumerate(collection):
            for u, v, p, q in collection[i+1:]:
                if max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q):
                    raise RuntimeError("I2VA canvas overlap")
    for x, y, w, h in boxes:
        if sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh
               for gx, gy, gw, gh in groups) != 1:
            raise RuntimeError("I2VA node outside group")
    if comfyui_root:
        verify_comfyui_nodes(comfyui_root, expected_types | {"MiniMaxH3ImageToVideo"}, set(),
                             [ROOT / "custom_nodes/minimax_h3_ordered_storyboard"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.workflow.read_text(encoding="utf-8")),
           json.loads(args.manifest.read_text(encoding="utf-8")), args.comfyui_root)
    print("I2VA first-frame route / assets / 2x and mosaic tail / layout verified")
