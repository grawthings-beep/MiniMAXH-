#!/usr/bin/env python3
"""Static contract for the imported FoxFur Seed Hunter workflow and models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_foxhunter_workflow import EROS, MYSTIC, UPSCALE, SOURCE_SHA256


REQUIRED_CUSTOM_TYPES = {
    "AnimeCadence", "MiniMaxH3MediaLoader", "MiniMaxH3ReferenceSplitter",
    "H3PromptIDE", "H3PromptReferenceInputs", "PixaromaLoraLoader",
    "PixaromaGroupSwitch", "PixaromaSwitch", "PixaromaSliders", "PixaromaLabel",
    "PixaromaText", "MiniMaxH3RefModsLoader", "MiniMaxH3RefModApply",
    "RIFEInterpolation", "Anything Everywhere", "Any Switch (rgthree)",
    "VHS_VideoCombine", "VHS_LoadAudioUpload", "easy seed", "easy ifElse",
    "MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward",
    "ModelPreviewOverrideKJ", "MiniMaxH3LatentUpscaleWithModel",
}


def validate(workflow: dict, manifest: dict) -> None:
    root = {n["id"]: n for n in workflow["nodes"]}
    definitions = {g["id"]: g for g in workflow["definitions"]["subgraphs"]}
    assert len(root) == 168 and len(definitions) == 6
    assert len(workflow["links"]) == 254
    assert all(link[1] in root and link[3] in root for link in workflow["links"])
    assert len({link[0] for link in workflow["links"]}) == len(workflow["links"])
    assert root[171]["type"] in definitions
    assert root[171]["widgets_values"][0] == EROS
    assert root[171]["widgets_values_named"]["unet_name"] == EROS
    unet = next(n for n in definitions[root[171]["type"]]["nodes"] if n["type"] == "UNETLoader")
    assert unet["widgets_values"][0] == EROS
    assert unet["widgets_values_named"]["unet_name"] == EROS
    assert root[605]["widgets_values"][0] == UPSCALE
    assert root[554]["widgets_values"][0]["loras"][0]["name"] == MYSTIC
    assert root[554]["widgets_values"][0]["loras"][0]["on"] is True
    assert root[554]["widgets_values"][0]["loras"][0]["sm"] == 0.5
    assert root[554]["widgets_values"][0]["loras"][0]["sc"] == 0.5
    assert root[554]["widgets_values_named"]["loras_ui"] == root[554]["widgets_values"][0]
    assert json.loads(root[554]["properties"]["loraLoaderState"]) == root[554]["widgets_values"][0]
    assert root[195]["mode"] == 0 and root[195]["widgets_values"] == [12, 7]
    assert root[195]["widgets_values_named"] == {"shift_video": 12, "shift_audio": 7}
    sliders = root[597]["properties"]["slidersState"]["sliders"]
    assert [slider["value"] for slider in sliders[:3]] == ["euler", "simple", 8]
    assert root[22]["widgets_values"][:3] == ["euler", "simple", 8]
    assert root[22]["widgets_values"][-1] == 5
    assert "Fox Fur Essence" not in root[365]["widgets_values"][0]
    assert workflow["extra"]["minimax_h3_foxhunter_import"]["source_sha256"] == SOURCE_SHA256
    all_types = {node["type"] for node in root.values()}
    for graph in definitions.values():
        all_types.update(node["type"] for node in graph["nodes"])
        node_ids = {node["id"] for node in graph["nodes"]}
        for link in graph["links"]:
            assert link["origin_id"] < 0 or link["origin_id"] in node_ids
            assert link["target_id"] < 0 or link["target_id"] in node_ids
    assert REQUIRED_CUSTOM_TYPES <= all_types, sorted(REQUIRED_CUSTOM_TYPES - all_types)
    files = {item["path"]: item for item in manifest["files"]}
    assert manifest["total_bytes"] == sum(item["size"] for item in files.values())
    assert len(files) == len(manifest["files"]) == 8
    for path in (
        "diffusion_models/" + EROS,
        "loras/" + MYSTIC,
        "latent_upscale_models/" + UPSCALE,
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "vae_approx/taeh3.safetensors",
        "rife/RIFEv4.26_0921.zip",
    ):
        assert path in files, path
    for item in files.values():
        assert len(item["sha256"]) == 64 and item["size"] > 0
        if "source_url" in item:
            assert item["source_url"].startswith("https://huggingface.co/")
            assert item.get("verify_sha256") is True
    model = files["diffusion_models/" + EROS]
    assert model["civitai_file_url"] == "https://civitai.com/api/download/models/3294059?fileId=3185144"
    assert model["size"] == 13_997_668_774
    assert model["sha256"] == "a8067999c65594b462d581c02b8a0573dd42d9812a14c586150a947c13388f2e"
    lora = files["loras/" + MYSTIC]
    assert lora["auth"] == "hf"
    assert lora["source_repo_id"] == "uwgm/illustrious"
    assert lora["source_path"] == "MysticXXX_MMH3-V5-ref2va.safetensors"
    assert lora["size"] == 138_339_432
    assert lora["sha256"] == "f57c76e761f7a5113e541ad58e84c648a5fe4d3cfbc277d0c44ae9763da29afe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    validate(json.loads(args.workflow.read_text(encoding="utf-8")),
             json.loads(args.manifest.read_text(encoding="utf-8")))
    print("[foxhunter] Eros beta5 / Mystic V5 workflow graph and eight model assets verified")


if __name__ == "__main__":
    main()
