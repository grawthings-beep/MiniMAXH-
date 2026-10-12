#!/usr/bin/env python3
"""CPU smoke: H3 nested mask, real uploaded MP4 and bit-identical video remux.

This deliberately does not load diffusion or VAE weights; GPU sound quality and
runtime must be checked on a Pod with the downloaded models.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def video_packet_hash(path):
    import av
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for packet in container.demux(video=0):
            if packet.size:
                digest.update(bytes(packet))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root.resolve()))
    sys.path.insert(0, str(ROOT / "custom_nodes"))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import av
    import folder_paths
    from comfy_api.latest import InputImpl
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from comfy_extras.nodes_minimax_h3 import video_latent_t
    from comfy_extras.nodes_video import CreateVideo
    import minimax_h3_ordered_storyboard as package
    from minimax_h3_ordered_storyboard import video_audio_nodes as foley
    from prepare_video_audio import build
    from verify_video_audio import verify

    torch.set_num_threads(2)
    assert foley.NODE_CLASS_MAPPINGS.keys() <= package.NODE_CLASS_MAPPINGS.keys()
    manifest, workflow = build("dasiwa-v2")
    verify(workflow, manifest, args.comfyui_root)
    tokenizer = MiniMaxH3Tokenizer()

    class Clip:
        def tokenize(self, prompt, **kwargs):
            refs = kwargs["minimax_ref_items"]
            assert len(refs) == 1 and refs[0]["type"] == "video"
            assert refs[0]["data"].shape[0] == 11  # 124 frames sampled at 2fps.
            return tokenizer.tokenize_with_weights(prompt, **kwargs)

        def encode_from_tokens_scheduled(self, tokens):
            assert tokens
            return [[torch.zeros(1, 1, 8), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]]

    class FakeVideoVAE:
        pass

    class VAE:
        first_stage_model = FakeVideoVAE()

        def encode(self, pixels):
            n, h, w, _ = pixels.shape
            return torch.full((1, 24, video_latent_t(n), h // 16, w // 16), .125)

    with tempfile.TemporaryDirectory(prefix="h3-audio-smoke-", dir=folder_paths.get_input_directory()) as temp:
        source_path = Path(temp) / "source.mp4"
        images = torch.zeros(120, 96, 64, 3)
        images[..., 0] = torch.arange(120)[:, None, None] / 120
        video = CreateVideo.execute(images, 24., None, 8).result[0]
        video.save_to(str(source_path), format="mp4", codec="h264")
        uploaded = InputImpl.VideoFromFile(str(source_path))
        source, report = foley.MiniMaxH3VideoAudioClip().prepare(uploaded, .1)
        assert source["length"] == 124 and source["frames"].shape == (124, 96, 64, 3), report
        assert source["duration"] == 5.0 and source["audio"]["waveform"].shape[-1] == 0
        prompt = foley.MiniMaxH3VideoAudioPrompt().build(foley.DEFAULT_AUDIO_PROMPT)[0]
        with patch("comfy.ldm.minimax.vae.MiniMaxH3VideoVAE", FakeVideoVAE):
            positive, latent = foley.MiniMaxH3VideoAudioCondition().encode(Clip(), VAE(), source, prompt)
        assert positive and latent["samples"].is_nested and latent["noise_mask"].is_nested
        locked_video, empty_audio = latent["samples"].unbind()
        video_mask, audio_mask = latent["noise_mask"].unbind()
        assert torch.all(locked_video == .125) and torch.count_nonzero(video_mask) == 0
        assert torch.all(audio_mask == 1) and empty_audio.shape[-1] == round(124/24*40)
        generated = {"waveform": torch.zeros(1, 2, round(5*32000)), "sample_rate": 32000}
        # Both an initially silent video and one with an old soundtrack must
        # end with the same original video packets and exactly one new audio track.
        original_audio = {"waveform": torch.ones(1, 2, 5*32000)*.25, "sample_rate": 32000}
        with_audio_path = Path(temp) / "source-with-audio.mp4"
        CreateVideo.execute(images, 24., original_audio, 8).result[0].save_to(
            str(with_audio_path), format="mp4", codec="h264")
        for current in (source_path, with_audio_path):
            current_source = dict(source, path=str(current))
            output = foley.MiniMaxH3VideoAudioSave().save(current_source, generated, "video/h3-audio-smoke")
            destination = Path(output.result[1])
            try:
                assert destination.is_file()
                assert video_packet_hash(current) == video_packet_hash(destination), "Original compressed frames changed"
                with av.open(str(destination)) as saved:
                    assert len(saved.streams.audio) == 1
                    assert sum(1 for _ in saved.decode(video=0)) == 120
            finally:
                destination.unlink(missing_ok=True)
    print("Video-to-audio CPU smoke passed: native H3 ref tokenizer, nested [video=0,audio=1] mask and bit-identical remux; diffusion quality/GPU not tested")


if __name__ == "__main__":
    main()
