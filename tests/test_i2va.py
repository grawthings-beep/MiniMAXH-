"""First-frame I2VA must stay distinct from free-composition R2VA and text-only T2VA."""
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_i2va import WORKFLOW, build, main as prepare_main
from prepare_r2v_profile import prepare, write_json
from verify_i2va import verify

package = types.ModuleType("i2va_unit_package")
package.__path__ = [str(ROOT / "custom_nodes/minimax_h3_ordered_storyboard")]
sys.modules[package.__name__] = package
i2va_nodes = importlib.import_module("i2va_unit_package.t2va_nodes")


def _node(workflow, kind):
    found = [n for n in workflow["nodes"] if n["type"] == kind]
    assert len(found) == 1
    return found[0]


def test_i2va_graph_uses_one_first_frame_and_same_hybrid_checkpoint():
    manifest, workflow = build("dasiwa-v2")
    verify(workflow, manifest)
    assert _node(workflow, "LoadImage")["widgets_values"][0] == ""
    assert _node(workflow, "MiniMaxH3FirstFrameToVideo")["title"].startswith("I2VA")
    assert _node(workflow, "SaveVideo")["widgets_values"][0].endswith("I2VA_First_Frame_2x")
    assert not any(n["type"] in {"MiniMaxH3Director", "MiniMaxH3ReferenceToVideo",
                                   "MiniMaxH3CharacterRefModR2V", "MiniMaxH3TextToVideo"}
                   for n in workflow["nodes"])
    baseline, _, _ = prepare("dasiwa-v2", vae_profile="int8")
    assert manifest["files"] == baseline["files"]


@pytest.mark.parametrize("profile", ["official", "dasiwa-turbo-v2"])
def test_i2va_refuses_unverified_mode_profile(profile):
    with pytest.raises(ValueError, match="dasiwa-v2"):
        build(profile)


def test_i2va_verifier_rejects_disconnected_image_and_early_mosaic():
    manifest, workflow = build("dasiwa-v2")
    cond = _node(workflow, "MiniMaxH3FirstFrameToVideo")
    broken = copy.deepcopy(workflow)
    _node(broken, "MiniMaxH3FirstFrameToVideo")["inputs"][2]["link"] = None
    with pytest.raises(RuntimeError):
        verify(broken, manifest)
    broken = copy.deepcopy(workflow)
    mosaic = _node(broken, "WanAutoMosaicVideo")
    link_id = mosaic["inputs"][0]["link"]
    next(link for link in broken["links"] if link[0] == link_id)[1] = _node(broken, "LoadImage")["id"]
    with pytest.raises(RuntimeError):
        verify(broken, manifest)
    assert cond["inputs"][2]["name"] == "first_frame"


def test_i2va_preset_merges_models_without_replacing_04(tmp_path, monkeypatch):
    baseline_manifest, _, baseline_04 = prepare("dasiwa-v2", vae_profile="x2-detail")
    write_json(tmp_path / "minimax_h3_r2v_models.json", baseline_manifest)
    write_json(tmp_path / "workflows/04_MiniMax_H3_Character_R2V_2x.json", baseline_04)
    original_04 = (tmp_path / "workflows/04_MiniMax_H3_Character_R2V_2x.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["prepare_i2va.py", "--profile", "dasiwa-v2", "--vae", "x2-detail",
                                   "--output-dir", str(tmp_path)])
    prepare_main()
    assert (tmp_path / "workflows/04_MiniMax_H3_Character_R2V_2x.json").read_bytes() == original_04
    assert (tmp_path / "workflows" / WORKFLOW).is_file()
    merged = json.loads((tmp_path / "minimax_h3_r2v_models.json").read_text(encoding="utf-8"))
    paths = {asset["path"] for asset in merged["files"]}
    assert "vae/minimax_h3_video_vae_int8_convrot.safetensors" in paths
    assert "upscale_models/RealESRGAN_x2plus.pth" in paths
    assert any("x2" in path for path in paths)


def test_first_frame_wrapper_calls_native_i2va_not_t2va(monkeypatch):
    # Exercise only the routing contract; a GPU generation is a separate RunPod check.
    MiniMaxH3FirstFrameToVideo = i2va_nodes.MiniMaxH3FirstFrameToVideo

    calls = []
    class Native:
        @staticmethod
        def execute(**kwargs):
            calls.append(kwargs)
            return ("conditioning", "latent")

    class Frame:
        shape = (1, 64, 64, 3)
        def __getitem__(self, key):
            return self

    class Clip:
        def tokenize(self, prompt, **kwargs):
            return {"a": [[(1, 1.0)]]}
        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3",
                        types.SimpleNamespace(MiniMaxH3ImageToVideo=Native))
    result = MiniMaxH3FirstFrameToVideo().encode(Clip(), "video-vae", Frame(),
                                                  "integrated_multimodal_description: A wave.", 448, 832, 124)
    assert result[:2] == ("conditioning", "latent")
    assert calls[0]["first_frame"].shape == (1, 64, 64, 3)
    assert calls[0]["last_frame"] is None
    assert calls[0]["vae"] == "video-vae"
    assert "I2VA" in result[2]
    with pytest.raises(ValueError, match="requires one uploaded"):
        MiniMaxH3FirstFrameToVideo().encode(Clip(), "video-vae", None, "A wave", 448, 832, 124)
