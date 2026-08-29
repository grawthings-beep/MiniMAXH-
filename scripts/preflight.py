#!/usr/bin/env python3
"""RunPod startup checks and a conservative H3 resolution recommendation."""

from __future__ import annotations

import argparse
import json
import os
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
        return {"name": "conservative-24gb", "megapixels": 0.4, "duration": 5}
    if vram_gib < 40:
        return {"name": "stable-32gb", "megapixels": 0.4, "duration": 5}
    return {"name": "high-48gb", "megapixels": 0.6, "duration": 5}


def query_gpu() -> tuple[str, float, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA GPU is not available to the container") from exc
    first = output.strip().splitlines()[0]
    name, memory_mib, driver_version = (part.strip() for part in first.rsplit(",", 2))
    try:
        vram_gib = float(memory_mib) / 1024
    except ValueError:
        # Some RunPod Community Cloud hosts expose the GPU but deny NVML's
        # memory.total query, returning "[Insufficient Permissions]". PyTorch
        # still knows the CUDA device allocation limit in that environment.
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch cannot access the NVIDIA GPU")
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            vram_gib = int(properties.total_memory) / GIB
            if not name or name.startswith("["):
                name = str(properties.name)
        except Exception as exc:
            raise RuntimeError(
                "NVIDIA VRAM could not be read through nvidia-smi or PyTorch"
            ) from exc
    return name, vram_gib, driver_version


def query_acceleration() -> dict[str, object]:
    result: dict[str, object] = {
        "ready": False,
        "gpu_access": False,
        "optimized": False,
        "torch_version": None,
        "compiled_cuda": None,
        "compute_capability": None,
        "comfy_kitchen_cuda": None,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["compiled_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            result["error"] = "PyTorch cannot access the NVIDIA GPU"
            return result
        result["gpu_access"] = True
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        result["compute_capability"] = f"{major}.{minor}"

        import comfy_kitchen as ck

        cuda = ck.list_backends().get("cuda", {})
        if not isinstance(cuda, dict):
            cuda = {"raw": str(cuda)}
        status = {
            "available": bool(cuda.get("available", False)),
            "disabled": bool(cuda.get("disabled", False)),
            "unavailable_reason": cuda.get("unavailable_reason"),
            "int8_capabilities": sorted(
                capability
                for capability in cuda.get("capabilities", [])
                if "int8" in capability or "convrot" in capability
            ),
        }
        result["comfy_kitchen_cuda"] = status
        result["optimized"] = status["available"] and not status["disabled"]
        result["ready"] = True
        if not result["optimized"]:
            result["warning"] = (
                "ComfyKitchen CUDA backend is unavailable or disabled; "
                "using the compatible eager path"
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


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
    parser.add_argument("--require-comfy-kitchen-cuda", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_bytes = int(manifest["total_bytes"])
    disk = shutil.disk_usage(existing_parent(args.model_dir))
    gpu_name, vram_gib, driver_version = query_gpu()
    ram_gib = memory_gib()
    profile = profile_for_vram(vram_gib)
    acceleration = query_acceleration()

    report = {
        "runtime_variant": os.environ.get(
            "MINIMAX_H3_RUNTIME_VARIANT", "unknown"
        ),
        "gpu": gpu_name,
        "nvidia_driver": driver_version,
        "vram_gib": round(vram_gib, 2),
        "ram_gib": round(ram_gib, 2),
        "disk_free_gib": round(disk.free / GIB, 2),
        "model_download_gib": round(model_bytes / GIB, 2),
        "recommended_profile": profile,
        "acceleration": acceleration,
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
    if not acceleration["gpu_access"]:
        print(
            "PyTorch cannot access this GPU with the selected container runtime. "
            f"Runtime variant: {report['runtime_variant']}. "
            "Use the community-cu128 image for r525-r579 hosts, or fast-cu130 "
            "only with an r580+ host driver.",
            file=sys.stderr,
        )
        return 1
    if args.require_comfy_kitchen_cuda and not acceleration["optimized"]:
        print(
            "ComfyKitchen CUDA acceleration is required but is not ready. "
            "Use the fast-cu130 image with an NVIDIA r580+ host driver, or select "
            "the community-cu128 image without the optimized-kernel requirement. "
            f"Details: {acceleration.get('error') or acceleration.get('warning')}",
            file=sys.stderr,
        )
        return 1
    if ram_gib < 30:
        print("WARNING: less than 32 GiB system RAM; offloading may be very slow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
