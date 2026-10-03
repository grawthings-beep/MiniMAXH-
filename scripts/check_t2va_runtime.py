"""CPU smoke: native H3 text tokenizer/latent, MP4 and 06 handoff, no diffusion weights."""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


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
    torch.set_num_threads(2)
    import av
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from comfy_extras.nodes_minimax_h3 import video_latent_t
    from comfy_extras.nodes_video import CreateVideo
    import minimax_h3_ordered_storyboard as package
    from minimax_h3_ordered_storyboard import t2va_nodes as text
    from minimax_h3_ordered_storyboard.swap_nodes import read_clip
    from prepare_t2va import build
    from verify_t2va import verify

    assert text.NODE_CLASS_MAPPINGS.keys() <= package.NODE_CLASS_MAPPINGS.keys()
    for cls in text.NODE_CLASS_MAPPINGS.values():
        assert isinstance(cls.INPUT_TYPES(), dict)
    manifest, workflow = build("dasiwa-v2")
    verify(workflow, manifest, args.comfyui_root)
    tokenizer = MiniMaxH3Tokenizer()  # Bundled vocab; no network or model weights.
    seen = []

    class Clip:
        def tokenize(self, prompt, **kwargs):
            assert kwargs == {"images": []}
            tokens = tokenizer.tokenize_with_weights(prompt, **kwargs)
            assert all(not isinstance(entry[0], dict) for batches in tokens.values() for batch in batches for entry in batch)
            return tokens

        def encode_from_tokens_scheduled(self, tokens):
            seen.append(tokens)
            # Qwen weights are fixtures; real tokenizer and native AV latent below.
            return [[torch.zeros(1, 1, 8), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]]

    # Two successive calls, different duration/aspect, no reference cache retained.
    for seconds, width, height, count in ((5., 480, 864, 124), (7., 640, 384, 175)):
        prompt, length = text.MiniMaxH3T2VAPrompt().build(seconds, text.DEFAULT_PROMPT)
        cond, latent, report = text.MiniMaxH3TextToVideo().encode(Clip(), prompt, width, height, length)
        assert not ({"minimax_keyframes", "minimax_refs"} & cond[0][1].keys())
        video, audio = latent["samples"].unbind()
        assert video.shape == (1, 24, video_latent_t(count), height//16, width//16)
        assert audio.shape == (1, 32, 2, round(count/24*40))
        assert f"{count} frames" in report
    assert len(seen) == 2
    try:
        text.MiniMaxH3TextToVideo().encode(Clip(), "x "*6000, 480, 864, 124)
    except ValueError as exc:
        assert "4096" in str(exc)
    else:
        raise AssertionError("Oversized prompt accepted")
    assert len(seen) == 2  # Rejected before text transformer.

    with tempfile.TemporaryDirectory(prefix="h3-t2va-smoke-") as temp:
        # Synthetic pixels/audio test the REAL native mux and 06 reader, not quality.
        frames = torch.zeros(124, 96, 64, 3)
        frames[:, :, :, 0] = torch.arange(124)[:, None, None] / 124
        sound = {"sample_rate": 32000, "waveform": torch.zeros(1, 2, round(124/24*32000))}
        video = CreateVideo.execute(frames, 24., sound, 8).result[0]
        path = Path(temp) / "text-master.mp4"
        video.save_to(str(path), format="mp4", codec="h264")
        with av.open(str(path)) as saved:
            assert len(saved.streams.audio) == 1
            assert float(saved.streams.video[0].average_rate) == 24
            assert sum(1 for _ in saved.decode(video=0)) == 124
        source = read_clip(path, 0., 5.2, .4)
        assert source["length"] == 124, "07 -> 06 must not lose an entire 17-frame block"
        assert source["frames"].shape == (124, 96, 64, 3)
        assert source["has_audio"]
    print("T2VA CPU smoke passed: real text-only tokenizer/native AV latent, MP4/124-frame 06 handoff; diffusion/VAE weights not run, GPU quality/OOM untested")


if __name__ == "__main__":
    main()
