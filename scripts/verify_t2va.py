"""Verify text-only routing, exact assets, decode guard and disjoint UI layout."""
import argparse
import json
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare
from verify_workflow import workflow_models, verify_comfyui_nodes


def verify(workflow, manifest, comfyui_root=None):
    nodes = workflow["nodes"]
    by_id, links = {n["id"]: n for n in nodes}, {l[0]: l for l in workflow["links"]}
    if len(by_id) != len(nodes) or len(links) != len(workflow["links"]):
        raise RuntimeError("Duplicate IDs")
    for lid, a, ai, b, bi, typ in links.values():
        if a not in by_id or b not in by_id:
            raise RuntimeError("Dangling link")
        out, inp = by_id[a]["outputs"][ai], by_id[b]["inputs"][bi]
        if out["type"] != typ or inp["type"] not in {typ, "*"} or inp["link"] != lid or out["links"].count(lid) != 1:
            raise RuntimeError("Serialized link mismatch")
        if by_id[a]["order"] >= by_id[b]["order"]:
            raise RuntimeError("Non-topological graph")
    for n in nodes:
        if n.get("mode", 0) != 0:
            raise RuntimeError("T2VA nodes must be active")
        for slot, inp in enumerate(n.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None and (lid not in links or links[lid][3:5] != [n["id"], slot]):
                raise RuntimeError("Dangling input")
        for slot, out in enumerate(n.get("outputs", [])):
            for lid in out.get("links") or []:
                if lid not in links or links[lid][1:3] != [n["id"], slot]:
                    raise RuntimeError("Dangling output")

    def one(kind):
        found = [n for n in nodes if n["type"] == kind]
        if len(found) != 1: raise RuntimeError(f"Expected one {kind}")
        return found[0]

    def route(target, inp_name, source, output=0):
        inp = next(i for i in target["inputs"] if i["name"] == inp_name)
        if inp.get("link") not in links or links[inp["link"]][1:3] != [source["id"], output]:
            raise RuntimeError(f"Incorrect {target['type']}.{inp_name} route")

    allowed = {"UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3SigmaShift", "BasicScheduler",
               "KSamplerSelect", "RandomNoise", "BasicGuider", "SamplerCustomAdvanced", "MiniMaxH3ReleaseVRAMLatent",
               "MiniMaxH3VAEDecodeTiled", "VAEDecodeAudio", "WanAutoMosaicVideo", "CreateVideo", "SaveVideo",
               "MiniMaxH3T2VAResolution", "MarkdownNote", "MiniMaxH3T2VAPrompt", "MiniMaxH3TextToVideo", "PreviewAny"}
    if {n["type"] for n in nodes} != allowed or workflow.get("definitions"):
        raise RuntimeError("Unexpected reference/LoRA/upscale/subgraph path")
    if manifest.get("r2v_profile") != "dasiwa-v2" or not manifest.get("t2va"):
        raise RuntimeError("T2VA must share DaSiWa v2")
    baseline_manifest, _, base = prepare("dasiwa-v2", vae_profile="int8")
    baseline = {n["type"]: n for n in base["nodes"]}
    for kind in ("UNETLoader", "CLIPLoader", "MiniMaxH3SigmaShift", "BasicScheduler", "KSamplerSelect", "WanAutoMosaicVideo"):
        if one(kind)["widgets_values"] != baseline[kind]["widgets_values"]:
            raise RuntimeError(f"Changed model/sampling/mosaic defaults: {kind}")
    expected = [f for f in baseline_manifest["files"] if not f["path"].startswith("upscale_models/")]
    if manifest["files"] != expected or manifest["total_bytes"] != sum(f["size"] for f in expected):
        raise RuntimeError("T2VA manifest differs from pinned shared assets")
    if workflow_models(nodes) != {f["path"] for f in expected} | {"auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"}:
        raise RuntimeError("Model dependencies differ from manifest")
    cond, prompt = one("MiniMaxH3TextToVideo"), one("MiniMaxH3T2VAPrompt")
    if prompt["widgets_values"][0] != 5. or "<Picture" in prompt["widgets_values"][1]:
        raise RuntimeError("Invalid image-free short default prompt")
    route(cond, "prompt", prompt)
    route(cond, "length", prompt, 1)
    route(cond, "clip", one("CLIPLoader"))
    for slot, name in enumerate(("width", "height")):
        route(cond, name, one("MiniMaxH3T2VAResolution"), slot)
    route(one("PreviewAny"), "source", cond, 2)
    if one("MiniMaxH3T2VAResolution")["widgets_values"] != ["9:16", .4]:
        raise RuntimeError("Invalid draft resolution")
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
    vaes = {n["widgets_values"][0]: n for n in nodes if n["type"] == "VAELoader"}
    if len(vaes) != 2: raise RuntimeError("Expected exactly two VAEs")
    for kind, part in (("MiniMaxH3VAEDecodeTiled", "video"), ("VAEDecodeAudio", "audio")):
        route(one(kind), "samples", guard)
        route(one(kind), "vae", next(n for name, n in vaes.items() if f"h3_{part}_vae" in name))
    route(one("WanAutoMosaicVideo"), "images", one("MiniMaxH3VAEDecodeTiled"))
    route(one("CreateVideo"), "images", one("WanAutoMosaicVideo"))
    route(one("CreateVideo"), "audio", one("VAEDecodeAudio"))
    route(one("SaveVideo"), "video", one("CreateVideo"))
    if one("CreateVideo")["widgets_values"][0] != 24:
        raise RuntimeError("H3 output must use 24fps")
    groups = [g["bounding"] for g in workflow["groups"]]
    boxes = [n["pos"] + n["size"] for n in nodes]
    for collection in (groups, boxes):
        for i, (x, y, w, h) in enumerate(collection):
            for u, v, p, q in collection[i+1:]:
                if max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q):
                    raise RuntimeError("Canvas overlap")
    for x, y, w, h in boxes:
        if sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh for gx, gy, gw, gh in groups) != 1:
            raise RuntimeError("Node outside group")
    if comfyui_root:
        verify_comfyui_nodes(comfyui_root, allowed | {"MiniMaxH3ImageToVideo"}, set(),
                             [ROOT / "custom_nodes/minimax_h3_ordered_storyboard"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.workflow.read_text(encoding="utf-8")), json.loads(args.manifest.read_text(encoding="utf-8")), args.comfyui_root)
    print("T2VA graph / assets / image-free routing / layout verified")
