#!/usr/bin/env python3
"""Extract only the verified RIFE weight from its pinned Hugging Face ZIP."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def extract(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        candidates = [entry for entry in bundle.infolist()
                      if not entry.is_dir() and Path(entry.filename).name == "flownet.pkl"]
        if len(candidates) != 1 or candidates[0].file_size < 1_000_000:
            raise ValueError("RIFE archive does not contain exactly one plausible flownet.pkl")
        entry = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        digest = hashlib.sha256()
        with bundle.open(entry) as source, temporary.open("wb") as destination:
            while chunk := source.read(4 * 1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
        if temporary.stat().st_size != entry.file_size:
            temporary.unlink(missing_ok=True)
            raise ValueError("RIFE extraction size mismatch")
        temporary.replace(target)
        print(f"[foxhunter] RIFE flownet.pkl ready: {target.stat().st_size} bytes, sha256={digest.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    extract(args.archive, args.target)


if __name__ == "__main__":
    main()
