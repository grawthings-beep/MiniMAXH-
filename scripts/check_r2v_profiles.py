#!/usr/bin/env python3
"""Check every profile against pinned native nodes; optionally execute CPU shifts."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from prepare_r2v_profile import ROOT, prepare, write_json
from prepare_upscale_compare import build as build_comparison
from verify_upscale_compare import verify as verify_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true", help="Execute real native nodes using PyTorch on CPU")
    args = parser.parse_args()
    settings = json.loads((ROOT / "manifests/r2v_profiles.json").read_text())["profiles"]
    for profile in settings:
        comparison_manifest, comparison_workflow = build_comparison(profile)
        verify_comparison(comparison_workflow, comparison_manifest, args.comfyui_root)
    with tempfile.TemporaryDirectory(prefix="h3-r2v-profiles-") as temp:
        for profile, vae in ((p, v) for p in settings for v in ("int8", "x2-detail")):
            manifest, _, workflow = prepare(profile, vae_profile=vae)
            manifest_path, workflow_path = Path(temp) / "manifest.json", Path(temp) / "workflow.json"
            write_json(manifest_path, manifest)
            write_json(workflow_path, workflow)
            subprocess.run([
                sys.executable, str(ROOT / "scripts/verify_workflow.py"),
                "--workflow", str(workflow_path), "--manifest", str(manifest_path),
                "--mode", "r2v", "--expect-x2-vae" if vae == "x2-detail" else "--expect-upscale",
                "--expect-auto-mosaic", "--expect-memory-safe-decode",
                "--auto-mosaic-manifest", str(ROOT / "manifests/auto_mosaic.json"),
                "--comfyui-root", str(args.comfyui_root),
                "--custom-node-root", str(ROOT / "custom_nodes/minimax_h3_ordered_storyboard"),
                "--custom-node-root", str(ROOT / "custom_nodes/minimax_h3_x2_vae"),
            ], check=True)
    if args.runtime:
        sys.path.insert(0, str(args.comfyui_root))
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        from types import SimpleNamespace
        from comfy.model_sampling import ModelSamplingAV
        from comfy.model_patcher import ModelPatcher
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        MiniMaxH3SigmaShift.define_schema()  # Real registration/schema, not text search only.
        model = torch.nn.Module()
        model.model_config = SimpleNamespace(sampling_settings={"shift": 12.0, "audio_shift": 3.0})
        model.model_sampling = ModelSamplingAV(model.model_config)
        original = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        for profile in ("dasiwa-v2", "dasiwa-turbo-v2", "dasiwa-v2"):
            s = settings[profile]
            shifted = MiniMaxH3SigmaShift.execute(original, s["shift_video"], s["shift_audio"]).result[0]
            sampling = shifted.get_model_object("model_sampling")
            assert sampling.shift == s["shift_video"] and sampling.audio_shift == s["shift_audio"]
            assert shifted.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] == s["shift_audio"]
            sigmas = BasicScheduler.execute(shifted, s["scheduler"], s["steps"], 1.0).result[0]
            assert len(sigmas) == s["steps"] + 1 and sigmas[-1].item() == 0
            assert torch.isfinite(sigmas).all() and torch.all(sigmas[:-1] >= sigmas[1:])
            assert not original.object_patches and model.model_sampling.shift == 12.0
        print("R2VA runtime: native sigma shift + scheduler passed on CPU for both DaSiWa profiles (not a GPU quality test)")


if __name__ == "__main__":
    main()
