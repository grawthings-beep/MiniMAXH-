from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
AUTO_FILES = tuple(sorted(path.name for path in WORKFLOWS.glob("*_auto_mosaic.json")))


def graphs(workflow: dict):
    return [workflow, *workflow.get("definitions", {}).get("subgraphs", [])]


def link_fields(link):
    if isinstance(link, dict):
        return (
            int(link["id"]),
            int(link["origin_id"]),
            int(link["origin_slot"]),
            int(link["target_id"]),
            int(link["target_slot"]),
            str(link["type"]),
        )
    return int(link[0]), int(link[1]), int(link[2]), int(link[3]), int(link[4]), str(link[5])


class AutoMosaicWorkflowTests(unittest.TestCase):
    def test_all_normal_workflows_remain_mosaic_free_and_all_derivatives_exist(self):
        normal = tuple(
            path for path in WORKFLOWS.glob("minimax_h3_*.json")
            if not path.name.endswith("_auto_mosaic.json")
            and not path.name.startswith("upstream_")
            and not path.name.startswith("minimax_h3_preset_")
        )
        self.assertEqual(len(normal), 12)
        self.assertEqual(len(AUTO_FILES), 12)
        for path in normal:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(
                any(node["type"] == "WanAutoMosaicVideo" for graph in graphs(workflow) for node in graph.get("nodes", [])),
                path.name,
            )

    def test_three_ui_presets_share_one_toggleable_mosaic_tail(self):
        presets = tuple(sorted(WORKFLOWS.glob("minimax_h3_preset_*.json")))
        self.assertEqual(len(presets), 3)
        for path in presets:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            mosaics = [
                node
                for graph in graphs(workflow)
                for node in graph.get("nodes", [])
                if node["type"] == "WanAutoMosaicVideo"
            ]
            self.assertEqual(len(mosaics), 1, path.name)
            self.assertIs(mosaics[0]["widgets_values"][1], True)

            for graph in graphs(workflow):
                nodes = [
                    node for node in graph.get("nodes", [])
                    if node.get("pos") and node.get("size")
                ]
                for index, left in enumerate(nodes):
                    lx, ly = map(float, left["pos"])
                    lw, lh = map(float, left["size"])
                    for right in nodes[index + 1:]:
                        rx, ry = map(float, right["pos"])
                        rw, rh = map(float, right["size"])
                        overlap = (
                            max(lx, rx) < min(lx + lw, rx + rw)
                            and max(ly, ry) < min(ly + lh, ry + rh)
                        )
                        self.assertFalse(overlap, (path.name, left["id"], right["id"]))

            subgraph = workflow["definitions"]["subgraphs"][0]
            model_group = next(
                group for group in subgraph["groups"] if group["title"] == "Models"
            )
            gx, gy, gw, gh = map(float, model_group["bounding"])
            acceleration = [
                node for node in subgraph["nodes"]
                if node["type"] in {
                    "ApplyMiniMaxH3FirstBlockCache",
                    "MiniMaxH3SigmaShift",
                    "MiniMaxH3TurboProfile",
                }
            ]
            for node in acceleration:
                x, y = map(float, node["pos"])
                width, height = map(float, node["size"])
                self.assertTrue(
                    gx <= x and gy <= y and x + width <= gx + gw and y + height <= gy + gh,
                    (path.name, node["id"]),
                )

    def test_auto_workflows_have_one_bidirectionally_serialized_final_node(self):
        for filename in AUTO_FILES:
            workflow = json.loads((WORKFLOWS / filename).read_text(encoding="utf-8"))
            matches = [
                (graph, node)
                for graph in graphs(workflow)
                for node in graph.get("nodes", [])
                if node["type"] == "WanAutoMosaicVideo"
            ]
            self.assertEqual(len(matches), 1, filename)
            graph, mosaic = matches[0]
            self.assertEqual(
                mosaic["widgets_values"],
                ["ntd11_anime_nsfw_segm_v5.pt", True, "JUST", 0.3, 0.5, 0, 3, "pussy,penis,testicles"],
            )
            self.assertNotIn("anus", mosaic["widgets_values"][-1])

            by_id = {int(node["id"]): node for node in graph["nodes"]}
            for raw_link in graph["links"]:
                link_id, origin_id, origin_slot, target_id, target_slot, _ = link_fields(raw_link)
                if origin_id < 0 or target_id < 0:  # subgraph virtual sockets
                    continue
                self.assertIn(origin_id, by_id, (filename, link_id))
                self.assertIn(target_id, by_id, (filename, link_id))
                self.assertIn(link_id, by_id[origin_id]["outputs"][origin_slot].get("links") or [])
                self.assertEqual(by_id[target_id]["inputs"][target_slot].get("link"), link_id)

            outgoing = [link_fields(link) for link in graph["links"] if link_fields(link)[1] == int(mosaic["id"])]
            incoming = [link_fields(link) for link in graph["links"] if link_fields(link)[3] == int(mosaic["id"])]
            self.assertEqual(len(incoming), 1, filename)
            self.assertEqual(len(outgoing), 1, filename)
            target = by_id[outgoing[0][3]]["type"]
            self.assertIn(target, {"CreateVideo", "MiniMaxH3StoryExport2x"})

    def test_auto_workflow_layout_has_no_node_or_group_overlap(self):
        def overlaps(left, right):
            lx, ly, lw, lh = left
            rx, ry, rw, rh = right
            return max(lx, rx) < min(lx + lw, rx + rw) and max(ly, ry) < min(ly + lh, ry + rh)

        for filename in AUTO_FILES:
            workflow = json.loads((WORKFLOWS / filename).read_text(encoding="utf-8"))
            for graph in graphs(workflow):
                nodes = [node for node in graph.get("nodes", []) if node.get("pos") and node.get("size")]
                for index, left in enumerate(nodes):
                    left_box = [*map(float, left["pos"]), *map(float, left["size"])]
                    for right in nodes[index + 1 :]:
                        right_box = [*map(float, right["pos"]), *map(float, right["size"])]
                        self.assertFalse(overlaps(left_box, right_box), (filename, left["id"], right["id"]))
                groups = graph.get("groups", [])
                for index, left in enumerate(groups):
                    for right in groups[index + 1 :]:
                        self.assertFalse(
                            overlaps(list(map(float, left["bounding"])), list(map(float, right["bounding"]))),
                            (filename, left.get("title"), right.get("title")),
                        )

    def test_static_verifier_covers_every_auto_variant(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_auto_mosaic_workflows.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified all 12", result.stdout)


if __name__ == "__main__":
    unittest.main()
