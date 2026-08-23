#!/usr/bin/env python3
"""Verify every published auto-mosaic derivative against its normal manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_workflow.py"
WORKFLOWS = ROOT / "workflows"
MANIFESTS = ROOT / "manifests"

CASES = (
    ("minimax_h3_i2v_auto_mosaic.json", "minimax_h3_i2v.json", "i2v", ()),
    ("minimax_h3_i2v_easycache_auto_mosaic.json", "minimax_h3_i2v.json", "i2v", ("--expect-easycache",)),
    ("minimax_h3_i2v_upscale_auto_mosaic.json", "minimax_h3_i2v_upscale.json", "i2v", ("--expect-upscale",)),
    ("minimax_h3_i2v_easycache_upscale_auto_mosaic.json", "minimax_h3_i2v_upscale.json", "i2v", ("--expect-easycache", "--expect-upscale")),
    ("minimax_h3_i2v_hmmotion_lora_upscale_auto_mosaic.json", "minimax_h3_i2v_upscale.json", "i2v", ("--expect-upscale", "--expect-lora")),
    (
        "minimax_h3_i2v_selectable_lora_upscale_auto_mosaic.json",
        "minimax_h3_i2v_upscale.json",
        "i2v",
        ("--expect-upscale", "--expect-lora", "HMNSFW_AIO_V2.safetensors", "--expect-lora-strength", "0.5"),
    ),
    ("minimax_h3_r2v_auto_mosaic.json", "minimax_h3_r2v.json", "r2v", ("--require-video-reference",)),
    ("minimax_h3_r2v_easycache_auto_mosaic.json", "minimax_h3_r2v.json", "r2v", ("--expect-easycache", "--require-video-reference")),
    ("minimax_h3_r2v_upscale_auto_mosaic.json", "minimax_h3_r2v_upscale.json", "r2v", ("--expect-upscale", "--require-video-reference")),
    ("minimax_h3_r2v_easycache_upscale_auto_mosaic.json", "minimax_h3_r2v_upscale.json", "r2v", ("--expect-easycache", "--expect-upscale", "--require-video-reference")),
    (
        "minimax_h3_story_quality_lora_2x_auto_mosaic.json",
        "minimax_h3_i2v_upscale.json",
        "story",
        ("--expect-upscale", "--expect-lora", "HMNSFW_AIO_V2.safetensors", "--expect-lora-strength", "0.5"),
    ),
    (
        "minimax_h3_story_easycache_lora_2x_auto_mosaic.json",
        "minimax_h3_i2v_upscale.json",
        "story",
        ("--expect-easycache", "--expect-upscale", "--expect-lora", "HMNSFW_AIO_V2.safetensors", "--expect-lora-strength", "0.5"),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--director-root", type=Path)
    parser.add_argument("--local-node-root", type=Path)
    args = parser.parse_args()
    for workflow, manifest, mode, extra in CASES:
        command = [
            sys.executable,
            str(VERIFY),
            "--workflow",
            str(WORKFLOWS / workflow),
            "--manifest",
            str(MANIFESTS / manifest),
            "--mode",
            mode,
            "--expect-auto-mosaic",
            "--auto-mosaic-manifest",
            str(MANIFESTS / "auto_mosaic.json"),
            *extra,
        ]
        if args.comfyui_root:
            command.extend(["--comfyui-root", str(args.comfyui_root)])
            if args.local_node_root:
                command.extend(["--custom-node-root", str(args.local_node_root)])
            if mode == "story" and args.director_root:
                command.extend(["--custom-node-root", str(args.director_root)])
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    print(f"Verified all {len(CASES)} auto-mosaic workflow variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
