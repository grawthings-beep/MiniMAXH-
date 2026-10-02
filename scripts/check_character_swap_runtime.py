"""Real CPU video/audio/native H3 contracts; diffusion/VAE weights are fixtures.

Optional real LoRA test maps every adapter to a meta-device H3 architecture.
This does NOT prove DaSiWa identity quality, GPU memory fit, or video fidelity.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def make_video(path, fps=30, seconds=3, audio=True, variable=False, rotation=0, offset=0.):
    import av
    import numpy as np
    with av.open(str(path), "w") as output:
        mp4 = path.suffix == ".mp4"
        stream = output.add_stream("libx264" if mp4 else "ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p" if mp4 else "bgr0"
        if rotation:
            stream.set_display_rotation(rotation)
        stream.time_base = Fraction(1, 1000)
        sound = output.add_stream("pcm_f32le", rate=44100) if audio else None
        if sound:
            sound.layout = "stereo"
        for i in range(round(seconds*fps)):
            timestamp = i / fps
            if variable and i % 2:
                timestamp += .009
            pixels = np.zeros((64, 96, 3), dtype=np.uint8)
            pixels[..., 0] = min(255, round(timestamp * 60))
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = round((timestamp+offset)*1000), Fraction(1, 1000)
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
        if sound:
            samples = np.ones((2, round(seconds*44100)), dtype=np.float32) * .15
            samples[:, 44100:] *= -1
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = 44100, 0, Fraction(1, 44100)
            output.mux(sound.encode(frame))
            output.mux(sound.encode())


def real_lora_test(path):
    import torch
    from safetensors.torch import load_file
    import comfy.lora
    import comfy.ops
    from comfy.model_base import MiniMaxH3
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_patcher import ModelPatcher
    from types import SimpleNamespace
    from minimax_h3_ordered_storyboard.turbo_nodes import MiniMaxH3R2VLoRA
    import folder_paths

    weights = load_file(str(path))
    model = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(unet_config={})
    with torch.device("meta"):
        model.diffusion_model = MiniMaxH3Model(device="meta", dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    patches = comfy.lora.load_lora(weights, key_map)
    loaded = set().union(*(p.loaded_keys for p in patches.values()))
    assert loaded == set(weights), f"Unmatched LoRA tensors: {set(weights)-loaded}"
    state = model.state_dict()
    for key, adapter in patches.items():
        up, down = adapter.weights[:2]
        assert tuple(state[key].shape) == (up.shape[0], down.shape[1]), key
        assert up.shape[1] == down.shape[0]
    folder_paths.add_model_folder_path("loras", str(path.parent))
    base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    node = MiniMaxH3R2VLoRA()
    patched = node.apply_lora(base, path.name, 1., True)[0]
    assert set(patched.patches) == set(patches) and not base.patches
    assert node.apply_lora(base, path.name, 1., False)[0] is base and node._loaded_lora is None
    print(f"Real pinned LoRA: all {len(weights)} tensors / {len(patches)} adapters match native H3 architecture; ON/OFF passes (meta weights, not quality)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--real-lora", action="store_true", help="Download/hash/test the 155MB adapter then delete it")
    parser.add_argument("--lora-file", type=Path, help="Already-downloaded real adapter")
    args = parser.parse_args()
    # Docker installs the pinned RefMod vendor into ComfyUI, not the source
    # checkout imported below. Local CPU fixtures can keep their explicit root.
    os.environ.setdefault("MINIMAX_H3_REFMOD_ROOT", str(
        args.comfyui_root.resolve() / "custom_nodes/minimax_h3_ordered_storyboard/refmod_vendor"))
    sys.path.insert(0, str(args.comfyui_root.resolve()))
    sys.path.insert(0, str(ROOT / "custom_nodes"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    import av
    import folder_paths
    from comfy_api.latest._input_impl.video_types import VideoFromFile
    from comfy_extras.nodes_video import CreateVideo
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy_extras.nodes_minimax_h3 import video_latent_t
    import minimax_h3_ordered_storyboard as package
    from minimax_h3_ordered_storyboard import swap_nodes as swap, refmod_nodes as ref
    from prepare_character_swap import build
    from verify_character_swap import verify

    assert swap.NODE_CLASS_MAPPINGS.keys() <= package.NODE_CLASS_MAPPINGS.keys()
    for p in ("official", "dasiwa-v2"):
        for v in ("int8", "x2-detail"):
            m, w = build(p, vae_profile=v)
            verify(w, m, args.comfyui_root)

    with tempfile.TemporaryDirectory(prefix="h3-swap-runtime-") as temp:
        temp = Path(temp)
        with patch.object(folder_paths, "get_input_directory", return_value=str(temp)):
            for fps, variable, sound in ((24, False, True), (30, False, True), (60, False, False), (30, True, True)):
                path = temp / f"source-{fps}-{variable}.mkv"
                make_video(path, fps=fps, audio=sound, variable=variable)
                # Full-movie decoding is forbidden even if available on VIDEO.
                with patch.object(VideoFromFile, "get_components", side_effect=AssertionError("Full video decoded")):
                    source, info = swap.MiniMaxH3SwapClip().prepare(VideoFromFile(str(path)), 1., 1.5, .4)
                assert source["length"] == 22 and source["frames"].shape == (22, 64, 96, 3)
                # Color encodes source timestamp. Verify real-time sampling, not
                # 30/60fps interpreted as 24fps slow motion, including VFR.
                colors = source["frames"][:, 0, 0, 0] * 255
                expected = (1 + torch.arange(22) / 24) * 60
                assert (colors-expected).abs().max() <= 4, (fps, colors)
                assert source["audio"]["waveform"].shape == (1, 2, 44000)
                assert source["has_audio"] == sound
                if sound:
                    assert abs(source["audio"]["waveform"][..., 1000:-1000].mean().item() + .15) < .001
                else:
                    assert torch.count_nonzero(source["audio"]["waveform"]) == 0
                assert "22 frames" in info
                short = swap.read_clip(path, 0., 5.2, .4)
                assert short["length"] == 56  # 3s source, never padded to requested 5s.
                for start, duration in ((10., 5.), (2.7, 5.), (0., float("nan"))):
                    try:
                        swap.read_clip(path, start, duration, .4)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Invalid/too-short clip accepted")
                generated = {"sample_rate": 24000, "waveform": torch.ones(1, 2, 20000)}
                for mode in ("source", "generated", "mute"):
                    actual = swap.MiniMaxH3SwapAudio().select(source, generated, mode)[0]
                    assert actual["waveform"].shape[-1] == round(22/24 * actual["sample_rate"])
                    if mode == "mute":
                        assert not actual["waveform"].any()
                # Real native CreateVideo -> H.264 MP4 with the trimmed soundtrack.
                video = CreateVideo.execute(source["frames"], 24., source["audio"], 8).result[0]
                target = temp / f"roundtrip-{fps}-{variable}.mp4"
                video.save_to(str(target), format="mp4", codec="h264")
                with av.open(str(target)) as saved:
                    assert len(saved.streams.audio) == 1
                    assert float(saved.streams.video[0].average_rate) == 24
                    assert sum(1 for _ in saved.decode(video=0)) == 22
            rotated = temp / "rotated.mp4"
            make_video(rotated, audio=False, rotation=90)
            rotated_clip = swap.read_clip(rotated, 0., 1.5, .4)
            assert rotated_clip["frames"].shape == (22, 96, 64, 3)
            delayed = temp / "delayed-video.mkv"
            make_video(delayed, offset=.5)
            delayed_clip = swap.read_clip(delayed, 0., 1.5, .4)
            # Audio begins .5s before the first video frame. Video-relative zero
            # must crop .5s off the audio, rather than resetting both separately.
            wave = delayed_clip["audio"]["waveform"]
            assert abs(wave[..., 1000:20000].mean().item() - .15) < .001
            assert abs(wave[..., 26000:42000].mean().item() + .15) < .001
            tokenizer = MiniMaxH3Tokenizer()
            seen = []

            class Clip:
                def tokenize(self, text, **kwargs):
                    seen.append(kwargs["minimax_ref_items"])
                    return tokenizer.tokenize_with_weights(text, **kwargs)

                def encode_from_tokens_scheduled(self, tokens):
                    return [[torch.zeros(1, 1, 8), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]]

            class VAE:
                first_stage_model = object.__new__(MiniMaxH3VideoVAE)

                def encode(self, image):
                    t = 1 if image.shape[0] == 1 else video_latent_t(image.shape[0])
                    return torch.zeros(1, 24, t, image.shape[1]//16, image.shape[2]//16)

            for count in (1, 3, 1):
                references = {"images": [torch.zeros(1, 64, 96, 3)]*count,
                              "names": [f"reference{i}.png" for i in range(count)], "budget": 2048}
                character, _ = ref.MiniMaxH3CreateCharacterRefMod().create(references, VAE())
                cond, latent = swap.MiniMaxH3CharacterSwap().encode(Clip(), character, source, VAE(), swap.DEFAULT_PROMPT)
                blocks = cond[0][1]["minimax_refs"]
                assert [b["kind"] for b in blocks] == ["image"]*count + ["video"]
                assert all(b["latent"].device.type == "cpu" for b in blocks)
                assert blocks[-1]["latent_t"] == video_latent_t(source["length"])
                assert [i["type"] for i in seen[-1]] == ["image"]*count + ["video"]
                assert seen[-1][-1]["timestamps"] == [0., .5]
                assert latent["samples"].is_nested
            for bad in ("<Video 2>", "<Audio 1>", "<Picture 2>", "x "*6000):
                try:
                    swap.MiniMaxH3CharacterSwap().encode(Clip(), character, source, VAE(), bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Missing reference/token budget accepted")
        if args.real_lora or args.lora_file:
            asset = json.loads((ROOT / "manifests/character_swap.json").read_text())["files"][0]
            path = args.lora_file or temp / Path(asset["path"]).name
            if not path.is_file():
                request = urllib.request.Request(asset["source_url"], headers={"User-Agent": "MiniMAXH-CPU-test"})
                with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
                    while chunk := response.read(1024*1024):
                        out.write(chunk)
            assert path.stat().st_size == asset["size"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
            real_lora_test(path)
    print("Character swap CPU smoke passed: real 24/30/60fps/VFR decode, trim/audio/MP4, native tokenizer + image/video RefMods; diffusion/VAE mocked, GPU/quality untested")


if __name__ == "__main__":
    main()
