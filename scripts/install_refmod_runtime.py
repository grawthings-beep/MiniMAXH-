"""Install only the upstream RefMod format/core from a verified source checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

COMMIT = "f9462081e28794389b5a6c5067eb327412ad8ee7"


def install(source, destination):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError("RefMod source commit mismatch")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "core.py", "bundle.py", "LICENSE"], text=True)
    if dirty.strip():
        raise ValueError("RefMod source files were modified")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("core.py", "bundle.py", "LICENSE"):
        shutil.copyfile(source / name, destination / name)
    (destination / "REVISION").write_text(COMMIT + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    install(args.source, args.destination)
