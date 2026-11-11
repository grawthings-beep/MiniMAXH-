#!/usr/bin/env python3
"""Fail the image build when a FoxFur workflow node did not register."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_foxhunter import REQUIRED_CUSTOM_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    args = parser.parse_args()
    registered = json.loads(args.object_info.read_text(encoding="utf-8"))
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    subgraph_ids = {graph["id"] for graph in workflow["definitions"]["subgraphs"]}
    needed = {node["type"] for node in workflow["nodes"]}
    for graph in workflow["definitions"]["subgraphs"]:
        needed.update(node["type"] for node in graph["nodes"])
    needed -= subgraph_ids | {"MarkdownNote", "Note"}
    missing = sorted(needed - registered.keys())
    if missing:
        raise SystemExit("[foxhunter] missing runtime node registrations: " + ", ".join(missing))
    assert REQUIRED_CUSTOM_TYPES <= registered.keys()
    print(f"[foxhunter] all {len(needed)} executable node IDs registered")


if __name__ == "__main__":
    main()
