"""Validate comparison wiring, model contracts and canvas geometry."""
import argparse
import json
from pathlib import Path

from prepare_r2v_profile import ROOT
from verify_workflow import verify_character_refmod, verify_dasiwa_profile, verify_comfyui_nodes, workflow_models


def verify(workflow, manifest, comfyui_root=None):
    nodes = workflow["nodes"]
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise RuntimeError("Duplicate node id")
    selected = {}
    for kind in ("MiniMaxH3CompareUpscale", "SamplerCustomAdvanced", "MiniMaxH3ReleaseVRAMLatent",
                 "VAEDecodeAudio", "ResolutionSelector", "MiniMaxH3CreateCharacterRefMod"):
        found = [n for n in nodes if n["type"] == kind and n.get("mode", 0) == 0]
        if len(found) != 1:
            raise RuntimeError(f"Expected one active {kind}")
        selected[kind] = found[0]
    if {n["type"] for n in nodes} & {"SaveVideo", "CreateVideo", "ImageUpscaleWithModel", "WanAutoMosaicVideo",
                                    "MiniMaxH3VAEDecodeTiled", "MiniMaxH3VAEDecodeFast", "UpscaleModelLoader"}:
        raise RuntimeError("Finishing must occur only inside the serial comparison node")
    links = {l[0]: l for l in workflow["links"]}
    if len(links) != len(workflow["links"]):
        raise RuntimeError("Duplicate link id")
    for lid, a, ai, b, bi, typ in links.values():
        if a not in by_id or b not in by_id:
            raise RuntimeError("Dangling link")
        output, input_ = by_id[a]["outputs"][ai], by_id[b]["inputs"][bi]
        if output["type"] != typ or input_["type"] != typ or input_["link"] != lid or lid not in output["links"]:
            raise RuntimeError("Serialized link/socket mismatch")
        if by_id[a]["order"] >= by_id[b]["order"]:
            raise RuntimeError("Non-topological comparison graph")
    for node in nodes:
        for index, inp in enumerate(node.get("inputs", [])):
            if inp.get("link") is not None and (inp["link"] not in links or links[inp["link"]][3:5] != [node["id"], index]):
                raise RuntimeError("Dangling serialized input")
        for index, out in enumerate(node.get("outputs", [])):
            for lid in out.get("links") or []:
                if lid not in links or links[lid][1:3] != [node["id"], index]:
                    raise RuntimeError("Dangling serialized output")
    stock = [n for n in nodes if n["type"] == "VAELoader" and n.get("widgets_values") == ["minimax_h3_video_vae_int8_convrot.safetensors"]]
    if len(stock) != 1:
        raise RuntimeError("Reference and stock decode must share one INT8 video VAE")
    compare = selected["MiniMaxH3CompareUpscale"]
    expected = [(selected["MiniMaxH3ReleaseVRAMLatent"]["id"], 0), (stock[0]["id"], 0),
                (selected["VAEDecodeAudio"]["id"], 0), (selected["ResolutionSelector"]["id"], 0),
                (selected["ResolutionSelector"]["id"], 1)]
    for inp, source in zip(compare["inputs"], expected):
        if tuple(links[inp["link"]][1:3]) != source:
            raise RuntimeError("Comparison must share sampled latent, audio and generation dimensions")
    sampler = selected["SamplerCustomAdvanced"]["id"]
    guard = selected["MiniMaxH3ReleaseVRAMLatent"]
    if links[guard["inputs"][0]["link"]][1:3] != [sampler, 0]:
        raise RuntimeError("Release barrier must receive the finished sampler output")
    audio = selected["VAEDecodeAudio"]
    if links[audio["inputs"][0]["link"]][1:3] != [guard["id"], 0]:
        raise RuntimeError("Audio must be decoded once after the release barrier")
    refs = selected["MiniMaxH3CreateCharacterRefMod"]
    vae_input = next(i for i in refs["inputs"] if i["name"] == "vae")
    if links[vae_input["link"]][1] != stock[0]["id"]:
        raise RuntimeError("Comparison reference encoder must use INT8")
    if compare["widgets_values"] != [480, 864, 24.0, "video/H3_Upscale_Compare", True]:
        raise RuntimeError("Comparison defaults changed")
    expected_models = {f["path"] for f in manifest["files"]} | {"auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"}
    if workflow_models(nodes) != expected_models:
        raise RuntimeError("Comparison model dependencies differ from manifest")
    for name in ("x2_detail_vae", "upscale_compare"):
        config = json.loads((ROOT / f"manifests/{name}.json").read_text(encoding="utf-8"))
        if any(f not in manifest["files"] for f in config["files"]):
            raise RuntimeError("Comparison assets must be pinned and hash-verified")
    verify_character_refmod(workflow)
    if manifest.get("r2v_profile", "official") != "official":
        verify_dasiwa_profile(workflow, manifest)
    boxes = [n["pos"] + n["size"] for n in nodes]
    groups = [g["bounding"] for g in workflow["groups"]]
    for collection in (boxes, groups):
        for i, (x, y, w, h) in enumerate(collection):
            for u, v, p, q in collection[i+1:]:
                if max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q):
                    raise RuntimeError("Comparison canvas overlap")
    for x, y, w, h in boxes:
        if sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh for gx, gy, gw, gh in groups) != 1:
            raise RuntimeError("Comparison node outside its group")
    if comfyui_root:
        verify_comfyui_nodes(comfyui_root, {n["type"] for n in nodes} | {"MiniMaxH3VAEDecodeFast", "UpscaleModelLoader", "ImageUpscaleWithModel"}, set(),
                             [ROOT / "custom_nodes/minimax_h3_ordered_storyboard", ROOT / "custom_nodes/minimax_h3_x2_vae"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.workflow.read_text(encoding="utf-8")), json.loads(args.manifest.read_text(encoding="utf-8")), args.comfyui_root)
    print("Comparison: wiring, same latent/audio/size, pinned models and layout passed")
