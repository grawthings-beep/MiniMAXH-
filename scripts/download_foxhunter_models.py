#!/usr/bin/env python3
"""Download public FoxHunter weights and the private LoRA in parallel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def public_manifest(manifest: dict) -> dict:
    public = {**manifest}
    public["files"] = [item for item in manifest["files"] if item.get("auth") != "hf"]
    public["total_bytes"] = sum(int(item["size"]) for item in public["files"])
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    private_assets = [item for item in manifest["files"] if item.get("auth") == "hf"]
    if len(private_assets) != 1:
        raise ValueError("FoxHunter requires exactly one private HF LoRA")
    private_asset = private_assets[0]
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN with access to uwgm/illustrious is required")
    # Fail before starting the large public downloads when the private mirror
    # cannot be read by this Pod's token.
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=token).model_info(
            private_asset["source_repo_id"], revision=private_asset["source_revision"], token=token
        )
        if private_asset["source_path"] not in {item.rfilename for item in info.siblings or []}:
            raise ValueError("LoRA file missing")
    except Exception:
        raise RuntimeError(
            "HF_TOKEN cannot read the requested private LoRA in uwgm/illustrious; "
            "public model downloads were not started"
        ) from None
    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="foxhunter-public-") as temporary:
        public_path = Path(temporary) / "manifest.json"
        public_path.write_text(json.dumps(public_manifest(manifest)), encoding="utf-8")
        public_env = os.environ.copy()
        public_env["MODEL_MANIFEST"] = str(public_path)
        public_env["COMFYUI_MODEL_DIR"] = str(args.root)
        public = subprocess.Popen(["bash", str(scripts / "download_models.sh")], env=public_env)
        private = subprocess.Popen([
            sys.executable, str(scripts / "download_foxhunter_hf.py"),
            "--manifest", str(args.manifest), "--root", str(args.root),
        ])
        public_status = public.wait()
        private_status = private.wait()
        if public_status or private_status:
            raise RuntimeError(
                f"FoxHunter downloads failed (public={public_status}, private={private_status}); "
                "ComfyUI will not start"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[foxhunter] {error}", file=sys.stderr)
        raise SystemExit(1) from None
