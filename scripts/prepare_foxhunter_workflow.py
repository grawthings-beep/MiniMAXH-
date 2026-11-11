#!/usr/bin/env python3
"""Import FoxFurEssence Seed Hunter v3 as a safe, portable RunPod preset.

The supplied JSON is data, not instructions. The imported graph is preserved;
only stale model paths, the author's sample prompt, preview metadata, and the
first-run duration are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "8c4df9a03003da0527a04ee915a00a65319aaa359b0e264edeaa6841264f4a94"
SINGULARITY = "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
UPSCALE = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
SILVER = "MINIMAX/minimax_h3_ref2v_turbo_silver_dareties_comfy_pruned_v2.safetensors"
NEUTRAL_PROMPT = """subject_definitions:
<Subject 1> is the same character shown in <Picture 1>. Preserve the reference character's face, hairstyle, outfit, proportions, and original rendering style.

summary:
A short, continuous character scene using the supplied reference picture.

retention_analysis:
Keep the character recognizable in every visible frame.

detailed_description:
The character notices the viewer, makes one clear, purposeful movement, and ends in a stable close shot. Keep the camera motion and physical reactions coherent. No extra people or cuts.

overall_soundscape:
Natural location ambience and a soft sound synchronized with the visible movement. No spoken dialogue.

non_diegetic_music:
N/A"""


def import_workflow(source: Path) -> dict:
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"Unexpected source JSON SHA256: {actual}")
    workflow = json.loads(raw)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    expected = {22: "b240b117-e18a-4bbf-bc5e-580a3284e8ac", 171: "fc72fed4-8bc4-4873-9164-c09cc633cce6", 365: "H3PromptIDE", 554: "PixaromaLoraLoader", 605: "LatentUpscaleModelLoader"}
    if any(nodes[node_id]["type"] != kind for node_id, kind in expected.items()):
        raise ValueError("Source graph layout changed; refusing to patch unknown nodes")
    if nodes[171]["widgets_values"][0] != SINGULARITY:
        raise ValueError("Source does not select the expected Singularity v1.3 model")
    nodes[22]["widgets_values"][-1] = 5
    nodes[365]["widgets_values"][0] = NEUTRAL_PROMPT
    nodes[605]["widgets_values"][0] = UPSCALE
    lora = nodes[554]["widgets_values"][0]["loras"]
    if len(lora) != 1 or not lora[0]["name"].replace("\\", "/").endswith(Path(SILVER).name):
        raise ValueError("Source Silver LoRA selection changed")
    lora[0]["name"] = SILVER
    for subgraph in workflow["definitions"]["subgraphs"]:
        if subgraph["id"] == nodes[171]["type"]:
            unet = next(node for node in subgraph["nodes"] if node["type"] == "UNETLoader")
            unet["widgets_values"][0] = SINGULARITY
    for node in workflow["nodes"]:
        if node["type"] == "VHS_VideoCombine":
            node["widgets_values"]["videopreview"] = {"hidden": False, "paused": False, "params": {}}
    workflow.setdefault("extra", {})["minimax_h3_foxhunter_import"] = {
        "source_sha256": SOURCE_SHA256,
        "first_run_seconds": 5,
        "model_profile": "singularity-v1.3-silver",
        "gpu_generation_verified": False,
    }
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workflow = import_workflow(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
