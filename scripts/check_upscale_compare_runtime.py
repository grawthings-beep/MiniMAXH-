"""Real CPU upscaler + MP4/AV test. VAE tensors are synthetic; no H3 quality claim."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]


def load_package(name, path):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Optional already-downloaded small upscale models")
    args = parser.parse_args()
    comfy_root = args.comfyui_root.resolve()
    sys.path.insert(0, str(comfy_root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    import nodes
    import folder_paths
    import av
    package = load_package("compare_runtime", ROOT / "custom_nodes/minimax_h3_ordered_storyboard")
    module = sys.modules["compare_runtime.compare_nodes"]
    cls = package.NODE_CLASS_MAPPINGS["MiniMaxH3CompareUpscale"]
    x2 = load_package("compare_x2_runtime", comfy_root / "custom_nodes/minimax_h3_x2_vae")
    nodes.NODE_CLASS_MAPPINGS.update(x2.NODE_CLASS_MAPPINGS)
    assert "MiniMaxH3VAEDecodeFast" in nodes.NODE_CLASS_MAPPINGS
    with tempfile.TemporaryDirectory(prefix="h3-compare-runtime-") as temp:
        temp = Path(temp)
        model_dir = args.model_dir.resolve() if args.model_dir else temp / "models"
        model_dir.mkdir(exist_ok=True)
        for asset in json.loads((ROOT / "manifests/upscale_compare.json").read_text())["files"]:
            path = model_dir / Path(asset["path"]).name
            if not path.is_file():
                request = urllib.request.Request(asset["source_url"], headers={"User-Agent": "MiniMAXH-CPU-test"})
                with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            assert path.stat().st_size == asset["size"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        folder_paths.add_model_folder_path("upscale_models", str(model_dir))
        folder_paths.set_output_directory(str(temp / "output"))
        audio = {"waveform": torch.zeros(1, 2, 24000), "sample_rate": 24000}
        samples = {"samples": torch.zeros(1)}
        stock = object()
        events, weak_frames = [], []

        def decode(actual, vae, label):
            assert actual is samples
            gc.collect()
            assert all(ref() is None for ref in weak_frames), "Previous branch pixels retained"
            size = 64 if label == "A_X2_VAE" else 32
            result = torch.linspace(0, 1, 3*size*size*3).reshape(3, size, size, 3)
            weak_frames.append(weakref.ref(result))
            events.append(label)
            return result

        saved = module._save_video
        def save(images, actual_audio, fps, path, metadata):
            assert actual_audio is audio
            assert images.device.type == "cpu" and images.shape == (3, 64, 64, 3)
            events.append("save")
            saved(images, actual_audio, fps, path, metadata)

        # Real upscaler loaders, CPU execution and H264/AAC encoders; only H3 weights mocked.
        with patch.object(nodes.VAELoader, "load_vae", return_value=(object(),)), \
             patch.object(module, "_decode", side_effect=decode), patch.object(module, "_save_video", side_effect=save):
            result = cls().compare(samples, stock, audio, 32, 32, 24, "video/test", False)
        report = json.loads(Path(result["result"][0]).read_text())
        assert report["status"] == "complete" and len(report["variants"]) == 3
        assert events == ["A_X2_VAE", "save", "B_INT8_SPAN", "save", "C_INT8_AnimeSharp", "save"]
        assert len(result["ui"]["comparison_files"]) == 3
        for row in report["variants"]:
            path = temp / "output" / row["subfolder"] / row["filename"]
            with av.open(str(path)) as video:
                assert video.streams.video[0].width == 64 and video.streams.video[0].height == 64
                assert len(video.streams.audio) == 1
                assert sum(1 for _ in video.decode(video=0)) == 3
            assert row["seconds"]["total"] > 0
        assert not list((temp / "output").rglob("*.partial.mp4"))
        # A successful branch survives a later error, with an explicit failed report.
        def fail_decode(actual, vae, label):
            if label == "B_INT8_SPAN":
                raise RuntimeError("synthetic failure")
            return torch.zeros(3, 64, 64, 3)
        with patch.object(nodes.VAELoader, "load_vae", return_value=(object(),)), patch.object(module, "_decode", side_effect=fail_decode):
            try:
                cls().compare(samples, stock, audio, 32, 32, 24, "video/failure", False)
            except RuntimeError as error:
                assert str(error) == "synthetic failure"
            else:
                raise AssertionError("Expected failure")
        failed = json.loads(next((temp / "output/video").glob("failure*_report.json")).read_text())
        assert failed["status"] == "failed" and len(failed["variants"]) == 1
        mosaic = sys.modules.get("compare_runtime.mosaic_nodes")
        with patch.object(mosaic.WanAutoMosaicVideo, "apply", return_value=("frames",)) as apply:
            assert module._mosaic("frames", True) == "frames"
            assert apply.call_args.args == ("frames", mosaic.MODEL_FILENAME, True, "JUST", .30, .50, 0, 3, "pussy,penis,testicles")
    print("CPU comparison passed: real SPAN/RCAN 2x, serial cleanup, three H264+audio MP4s, failure report. H3 VAE/GPU quality untested.")


if __name__ == "__main__":
    main()
