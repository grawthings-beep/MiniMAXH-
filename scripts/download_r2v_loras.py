#!/usr/bin/env python3
"""Opt-in R2VA creator downloads; reuse the pinned, verified legacy transports."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

SCRIPTS = {
    "hmmotion_v1": "download_lora.py",
    "hmnsfw_aio_v2": "download_civitai_lora.py",
    "mysticxxx_v4": "download_extra_loras.py",
}
MYSTIC_URL = "https://huggingface.co/uwgm/nikke-loras/resolve/main/MysticXXX_MMH3-V4.safetensors"
LEGACY_IDS = {"hmmotion_v1", "hmnsfw_aio_v2"}


def selection(env: dict) -> tuple[str, ...]:
    raw = env.get("H3_R2V_LORA_SELECTION", "none").strip().lower()
    if raw in {"", "none"}:
        return ()
    ids = set(SCRIPTS) if raw == "all" else {s.strip() for s in raw.split(",")}
    if not ids <= SCRIPTS.keys():
        raise ValueError("H3_R2V_LORA_SELECTION must be none, all, or a comma-separated list of: " + ", ".join(sorted(SCRIPTS)))
    return tuple(sorted(ids))


def download_env(env: dict) -> tuple[tuple[str, ...], dict]:
    ids = selection(env)
    child = dict(env)
    child["H3_LORA_SELECTION"] = ",".join(i for i in ids if i in LEGACY_IDS) or "none"
    # Explicitly requested files are required, irrespective of stale legacy flags.
    child["H3_LORA_REQUIRED"] = "1"
    child["CIVITAI_TOKEN"] = env.get("CIVITAI_API_TOKEN", "").strip() or env.get("CIVITAI_TOKEN", "").strip()
    if (set(ids) & {"hmmotion_v1", "mysticxxx_v4"}) and not env.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required for selected private Hugging Face LoRAs")
    if "hmnsfw_aio_v2" in ids and not child["CIVITAI_TOKEN"]:
        raise ValueError("CIVITAI_API_TOKEN (or CIVITAI_TOKEN) is required for the selected V2 LoRA")
    return ids, child


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate selection and credentials without downloads")
    args = parser.parse_args()
    ids, env = download_env(os.environ)
    print("[r2v-lora] startup selection: " + (", ".join(ids) or "none"), flush=True)
    if args.check or not ids:
        return 0
    processes = []
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        for lora_id in ids:
            command = [sys.executable, str(Path(__file__).with_name(SCRIPTS[lora_id]))]
            if lora_id == "mysticxxx_v4":
                command.extend(["--url", MYSTIC_URL, "--required"])
            processes.append(subprocess.Popen(command, env=env))
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("A selected R2VA LoRA failed download/integrity validation; ComfyUI will not start")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print("[r2v-lora] selected LoRAs verified; UI remains OFF until enabled")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError):
        # No transport URLs, credentials or arbitrary exception payloads in logs.
        print("[r2v-lora] failed: check H3_R2V_LORA_SELECTION, secret access and verified download messages", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
