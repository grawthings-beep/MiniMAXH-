"""Validate replacement routing, finishing, assets, and non-overlapping layout."""
import argparse
import json
from pathlib import Path
from prepare_r2v_profile import ROOT, prepare
from prepare_character_swap import LORA
from verify_workflow import workflow_models, verify_comfyui_nodes


def verify(workflow, manifest, comfyui_root=None):
    nodes = workflow["nodes"]
    by_id = {n["id"]: n for n in nodes}
    links = {l[0]: l for l in workflow["links"]}
    if len(by_id) != len(nodes) or len(links) != len(workflow["links"]):
        raise RuntimeError("Duplicate IDs")
    for lid, a, ai, b, bi, typ in links.values():
        if a not in by_id or b not in by_id:
            raise RuntimeError("Dangling link")
        out, inp = by_id[a]["outputs"][ai], by_id[b]["inputs"][bi]
        if out["type"] != typ or inp["type"] not in {typ, "*"} or inp["link"] != lid or lid not in out["links"]:
            raise RuntimeError("Serialized link mismatch")
        if by_id[a]["order"] >= by_id[b]["order"]:
            raise RuntimeError("Non-topological graph")
    for node in nodes:
        for index, inp in enumerate(node.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None and (lid not in links or links[lid][3:5] != [node["id"], index]):
                raise RuntimeError("Dangling input")
        for index, out in enumerate(node.get("outputs", [])):
            for lid in out.get("links") or []:
                if lid not in links or links[lid][1:3] != [node["id"], index]:
                    raise RuntimeError("Dangling output")

    def one(kind):
        found = [n for n in nodes if n["type"] == kind and n.get("mode", 0) == 0]
        if len(found) != 1:
            raise RuntimeError(f"Expected exactly one active {kind}")
        return found[0]

    def route(target, input_name, source, slot=0):
        inp = next(i for i in target["inputs"] if i["name"] == input_name)
        if inp.get("link") not in links or links[inp["link"]][1:3] != [source["id"], slot]:
            raise RuntimeError(f"Incorrect {target['type']}.{input_name} route")

    source, cond, refs = one("MiniMaxH3SwapClip"), one("MiniMaxH3CharacterSwap"), one("MiniMaxH3CreateCharacterRefMod")
    prompt, lora = one("MiniMaxH3SwapPrompt"), one("MiniMaxH3R2VLoRA")
    route(source, "video", one("LoadVideo"))
    route(refs, "references", one("MiniMaxH3RefModImages"))
    route(cond, "character", refs)
    route(cond, "source", source)
    route(cond, "clip", one("CLIPLoader"))
    route(cond, "prompt", prompt)
    target, finish = one("MiniMaxH3SwapTarget"), one("MiniMaxH3SwapFinish")
    route(target, "source", source)
    route(cond, "edit", target)
    route(finish, "source", source)
    route(finish, "edit", target)
    if target["widgets_values"] != ["masked_replace", .25, 32, .1, 48] or cond["widgets_values"] != [""]:
        raise RuntimeError("Masked workflow requires fresh mask approval and bounded CPU defaults")
    if finish["widgets_values"] != [True, .985]:
        raise RuntimeError("Source-copy rejection must default ON")
    stock = [n for n in nodes if n["type"] == "VAELoader" and n["widgets_values"] == ["minimax_h3_video_vae_int8_convrot.safetensors"]]
    if len(stock) != 1:
        raise RuntimeError("Exactly one INT8 reference VAE required")
    route(cond, "vae", stock[0])
    route(refs, "vae", stock[0])
    sampler, barrier = one("SamplerCustomAdvanced"), one("MiniMaxH3ReleaseVRAMLatent")
    route(sampler, "latent_image", cond, 1)
    route(one("BasicGuider"), "conditioning", cond)
    route(barrier, "samples", sampler)
    audio, selection = one("VAEDecodeAudio"), one("MiniMaxH3SwapAudio")
    route(audio, "samples", barrier)
    route(selection, "generated", audio)
    route(selection, "source", source)
    route(one("CreateVideo"), "audio", selection)
    route(one("CreateVideo"), "images", one("WanAutoMosaicVideo"))
    route(one("SaveVideo"), "video", one("CreateVideo"))
    if selection["widgets_values"] != ["source"] or source["widgets_values"] != [0., 5.2, .4]:
        raise RuntimeError("Unexpected clip/audio defaults")
    if lora["widgets_values"] != [LORA, 1.0, True]:
        raise RuntimeError("Character swap LoRA must default ON / 1.0")
    if any("Turbo" in n["type"] or n["type"] in {"MiniMaxH3FullPrompt", "MiniMaxH3CharacterRefModR2V", "MiniMaxH3CompareUpscale"} for n in nodes):
        raise RuntimeError("Wrong conditioning/extra sampler path")
    profile = manifest.get("r2v_profile", "official")
    baseline = prepare(profile, vae_profile=manifest.get("vae_profile", "int8"))[2]
    baseline_nodes = {n["type"]: n for n in baseline["nodes"]}
    for kind in ("UNETLoader", "MiniMaxH3SigmaShift") if profile != "official" else ("UNETLoader",):
        if one(kind)["widgets_values"] != baseline_nodes[kind]["widgets_values"]:
            raise RuntimeError("Selected base model/shift was changed")
    model, guider, scheduler = one("UNETLoader"), one("BasicGuider"), one("BasicScheduler")
    route(lora, "model", model)
    tail = one("MiniMaxH3SigmaShift") if profile != "official" else lora
    if tail is not lora:
        route(tail, "model", lora)
    route(guider, "model", tail)
    route(scheduler, "model", tail)
    if scheduler["widgets_values"] != ["simple", 20, 1.0] or one("KSamplerSelect")["widgets_values"] != ["res_multistep"]:
        raise RuntimeError("Replacement sampling defaults must use 20 steps")
    decoder = one("MiniMaxH3VAEDecodeFast" if manifest.get("vae_profile") == "x2-detail" else "MiniMaxH3VAEDecodeTiled")
    route(decoder, "samples", barrier)
    finishing = decoder
    if manifest.get("vae_profile") != "x2-detail":
        finishing = one("ImageUpscaleWithModel")
        route(finishing, "image", decoder)
    route(finish, "images", finishing)
    route(one("WanAutoMosaicVideo"), "images", finish)
    if workflow_models(nodes) != {f["path"] for f in manifest["files"]} | {"auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"}:
        raise RuntimeError("Model dependencies differ from manifest")
    pinned = json.loads((ROOT / "manifests/character_swap.json").read_text())["files"][0]
    if pinned not in manifest["files"]:
        raise RuntimeError("Replacement LoRA must be pinned / SHA256 checked")
    detector = json.loads((ROOT / "manifests/swap_guard.json").read_text())["files"][0]
    if detector not in manifest["files"]:
        raise RuntimeError("Person segmentation model must be pinned / SHA256 checked")
    groups, boxes = [g["bounding"] for g in workflow["groups"]], [n["pos"] + n["size"] for n in nodes]
    for collection in (groups, boxes):
        for i, (x, y, w, h) in enumerate(collection):
            for u, v, p, q in collection[i+1:]:
                if max(x, u) < min(x+w, u+p) and max(y, v) < min(y+h, v+q):
                    raise RuntimeError("Canvas overlap")
    for x, y, w, h in boxes:
        if sum(gx <= x and gy <= y-25 and x+w <= gx+gw and y+h <= gy+gh for gx, gy, gw, gh in groups) != 1:
            raise RuntimeError("Node outside group")
    if comfyui_root:
        verify_comfyui_nodes(comfyui_root, {n["type"] for n in nodes}, set(),
                             [ROOT / "custom_nodes/minimax_h3_ordered_storyboard", ROOT / "custom_nodes/minimax_h3_x2_vae"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.workflow.read_text(encoding="utf-8")), json.loads(args.manifest.read_text(encoding="utf-8")), args.comfyui_root)
    print("Character swap graph / pinned assets / layout verified")
