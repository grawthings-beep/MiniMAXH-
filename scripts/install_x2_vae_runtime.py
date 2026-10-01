"""Install the creator-required decoder and its helper from a pinned checkout."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def install(source: Path, destination: Path):
    config = json.loads((ROOT / "manifests/x2_detail_vae.json").read_text(encoding="utf-8"))
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != config["decoder_commit"]:
        raise ValueError("X2 VAE decoder source commit mismatch")
    files = ("vae_decode.py", "utils.py", "README.md")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", *files], text=True)
    if dirty.strip():
        raise ValueError("X2 VAE decoder source was modified")
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copyfile(source / name, destination / name)
    # Preserve any license supplied at the pinned revision; do not invent one.
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    shutil.copyfile(ROOT / "custom_nodes/minimax_h3_x2_vae/__init__.py", destination / "__init__.py")
    (destination / "REVISION").write_text(revision + "\n", encoding="utf-8")
    (destination / "SOURCE").write_text(config["decoder_repository"] + "/tree/" + revision + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    install(args.source, args.destination)
