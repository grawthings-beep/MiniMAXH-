"""CPU smoke of the real native H3 first-frame conditioning path; no model weights."""
import argparse
from pathlib import Path
import sys

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
    import minimax_h3_ordered_storyboard as package
    from minimax_h3_ordered_storyboard.t2va_nodes import MiniMaxH3FirstFrameToVideo, MiniMaxH3I2VAPrompt
    from prepare_i2va import build
    from verify_i2va import verify

    manifest, workflow = build("dasiwa-v2")
    verify(workflow, manifest, args.comfyui_root)
    assert {"MiniMaxH3FirstFrameToVideo", "MiniMaxH3I2VAPrompt"} <= package.NODE_CLASS_MAPPINGS.keys()

    class Clip:
        def tokenize(self, prompt, **kwargs):
            assert len(kwargs["images"]) == 1
            assert kwargs["images"][0].shape == (1, 832, 448, 3)
            return {"a": [[(1, 1.0)]]}

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 1, 8), {}]]

    class VAE:
        def encode(self, pixels):
            assert pixels.shape == (1, 832, 448, 3)
            return torch.zeros(1, 24, 2, 52, 28)

    prompt, length = MiniMaxH3I2VAPrompt().build(5.0, "integrated_multimodal_description: The character waves.")
    assert length == 124
    frame = torch.zeros(1, 128, 64, 3)
    cond, latent, report = MiniMaxH3FirstFrameToVideo().encode(Clip(), VAE(), frame, prompt, 448, 832, length)
    assert len(cond[0][1]["minimax_keyframes"]) == 1
    assert cond[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert latent["samples"].tensors[0].shape[3:] == (52, 28)
    assert "first supplied image is frame 0" in report
    print("I2VA CPU smoke passed: first-frame image reaches native H3 keyframe zero; diffusion/VAE weights and GPU untested")


if __name__ == "__main__":
    main()
