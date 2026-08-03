#!/usr/bin/env python3
"""RunPod startup checks and a conservative H3 I2V resolution recommendation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


GIB = 1024**3
MODEL_HEADROOM = 8 * GIB


def profile_for_vram(vram_gib: float) -> dict[str, object]:
    if vram_gib < 8:
        return {"name": "unsupported", "megapixels": 0.0, "duration": 0}
    if vram_gib < 16:
        return {"name": "preview", "megapixels": 0.2, "duration": 5}
    if vram_gib < 24:
        return {"name": "balanced", "megapixels": 0.4, "duration": 5}
    if vram_gib < 32:
        return {"name": "high", "megapixels": 0.6, "duration": 5}
    return {"name": "native-768p", "megapixels": 0.98, "duration": 5}


def query_gpu() -> tuple[str, float]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA GPU is not available to the container") from exc
    first = output.strip().splitlines()[0]
    name, memory_mib = (part.strip() for part in first.rsplit(",", 1))
    return name, float(memory_mib) / 1024


def memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("Could not read system memory")


def existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"No existing parent for {path}")
        candidate = candidate.parent
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_bytes = int(manifest["total_bytes"])
    disk = shutil.disk_usage(existing_parent(args.model_dir))
    gpu_name, vram_gib = query_gpu()
    ram_gib = memory_gib()
    profile = profile_for_vram(vram_gib)

    report = {
        "gpu": gpu_name,
        "vram_gib": round(vram_gib, 2),
        "ram_gib": round(ram_gib, 2),
        "disk_free_gib": round(disk.free / GIB, 2),
        "model_download_gib": round(model_bytes / GIB, 2),
        "recommended_profile": profile,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    required = model_bytes + MODEL_HEADROOM
    if disk.free < required:
        print(
            f"At least {required / GIB:.1f} GiB free is required in the writable layer.",
            file=sys.stderr,
        )
        return 1
    if profile["name"] == "unsupported":
        print("At least 8 GiB VRAM is required; 12 GiB or more is recommended.", file=sys.stderr)
        return 1
    if ram_gib < 30:
        print("WARNING: less than 32 GiB system RAM; offloading may be very slow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

