from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "minimax_h3_story_workflow_builder_test",
    ROOT / "scripts" / "build_story_workflows.py",
)
storyboard = load_module(
    "minimax_h3_storyboard_schema_test",
    ROOT / "custom_nodes" / "minimax_h3_ordered_storyboard" / "storyboard.py",
)
exporter = load_module(
    "minimax_h3_story_exporter_schema_test",
    ROOT / "custom_nodes" / "minimax_h3_ordered_storyboard" / "exporter.py",
)


class StoryWorkflowTests(unittest.TestCase):
    def load_workflow(self, filename: str) -> dict:
        return json.loads((ROOT / "workflows" / filename).read_text(encoding="utf-8"))

    def test_quality_and_fast_graphs_pass_static_link_validation(self) -> None:
        quality = self.load_workflow(builder.QUALITY_OUTPUT)
        fast = self.load_workflow(builder.FAST_OUTPUT)
        builder.validate_generated(quality, fast=False)
        builder.validate_generated(fast, fast=True)

    def test_workflow_sockets_match_registered_custom_nodes(self) -> None:
        runtime_story_inputs = list(
            storyboard.MiniMaxH3OrderedStoryboard.INPUT_TYPES()["required"]
        )
        runtime_export_types = exporter.MiniMaxH3StoryExport2x.INPUT_TYPES()
        runtime_export_inputs = [
            *runtime_export_types["required"],
            *runtime_export_types["optional"],
        ]
        for filename in (builder.QUALITY_OUTPUT, builder.FAST_OUTPUT):
            workflow = self.load_workflow(filename)
            story_node = next(
                node
                for node in workflow["nodes"]
                if node["type"] == "MiniMaxH3OrderedStoryboard"
            )
            export_node = next(
                node
                for node in workflow["nodes"]
                if node["type"] == "MiniMaxH3StoryExport2x"
            )
            self.assertEqual(
                [item["name"] for item in story_node["inputs"]], runtime_story_inputs
            )
            self.assertEqual(
                [item["name"] for item in story_node["outputs"]],
                list(storyboard.MiniMaxH3OrderedStoryboard.RETURN_NAMES),
            )
            self.assertEqual(
                [item["name"] for item in export_node["inputs"]], runtime_export_inputs
            )
            self.assertEqual(
                [item["name"] for item in export_node["outputs"]],
                list(exporter.MiniMaxH3StoryExport2x.RETURN_NAMES),
            )

    def test_defaults_match_backend_contract(self) -> None:
        self.assertEqual(builder.STORYBOARD_STATE, storyboard.DEFAULT_STATE)
        runtime = exporter.MiniMaxH3StoryExport2x.INPUT_TYPES()["required"]
        expected_export_widgets = [
            runtime["fps"][1]["default"],
            "video/MiniMax_H3_Story_Quality_LoRA_2x",
            runtime["crf"][1]["default"],
            runtime["preset"][1]["default"],
            runtime["drop_boundary_duplicates"][1]["default"],
            runtime["drop_loop_terminal"][1]["default"],
        ]
        quality = self.load_workflow(builder.QUALITY_OUTPUT)
        export_node = next(
            node for node in quality["nodes"] if node["type"] == "MiniMaxH3StoryExport2x"
        )
        self.assertEqual(export_node["widgets_values"], expected_export_widgets)
        self.assertTrue(exporter.MiniMaxH3StoryExport2x.INPUT_IS_LIST)
        self.assertTrue(exporter.MiniMaxH3StoryExport2x.OUTPUT_NODE)


if __name__ == "__main__":
    unittest.main()
