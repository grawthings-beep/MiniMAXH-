"""CPU contract smoke: real native H3/tokenizer + RefMod, with model-weight fixtures.

Does not claim a real GPU generation, VAE reconstruction, identity, or OOM test.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.comfyui_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "custom_nodes"))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from PIL import Image
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    import minimax_h3_ordered_storyboard as package
    from minimax_h3_ordered_storyboard import refmod_nodes as ref
    import folder_paths

    assert ref.NODE_CLASS_MAPPINGS.keys() <= package.NODE_CLASS_MAPPINGS.keys()
    for cls in ref.NODE_CLASS_MAPPINGS.values():
        assert isinstance(cls.INPUT_TYPES(), dict)
    core, bundle = ref.refmod_modules()
    tokenizer = MiniMaxH3Tokenizer()  # Packaged vocabulary, no network/model weights.

    class Clip:
        def tokenize(self, text, **kwargs):
            return tokenizer.tokenize_with_weights(text, **kwargs)

        def encode_from_tokens_scheduled(self, tokens):
            # Validate actual tokenizer payload. Transformer weights are NOT loaded.
            entries = tokens["qwen3vl_32b"][0]
            assert any(isinstance(t[0], dict) and t[0]["type"] == "image" for t in entries)
            return [[torch.zeros(1, 1, 8), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]]

    class VAE:
        first_stage_model = object.__new__(MiniMaxH3VideoVAE)

        def encode(self, image):
            return torch.ones(1, 24, 1, image.shape[1] // 16, image.shape[2] // 16)

    with tempfile.TemporaryDirectory(prefix="h3-refmod-smoke-") as temp:
        folder = Path(temp)
        Image.new("RGB", (256, 384), "blue").save(folder / "a.png")
        Image.new("RGB", (512, 256), "red").save(folder / "b.png")
        with mock.patch.object(folder_paths, "get_input_directory", return_value=temp), \
             mock.patch.object(folder_paths, "get_output_directory", return_value=temp):
            for names in (["a.png"], ["a.png", "b.png"], ["b.png"]):
                images = ref.MiniMaxH3RefModImages().load(json.dumps(names), 1024, 2048)[0]
                character, mods = ref.MiniMaxH3CreateCharacterRefMod().create(images, VAE())
                for prompt in (ref.DEFAULT_FULL_PROMPT, "```text\n<Subject 1> from <Picture 1> waves.\n```",
                               "元画像と同じ顔。A fast movement, then pause. (gentle:1.2) {literal text}"):
                    text, length = ref.MiniMaxH3FullPrompt().build(5, prompt)
                    cond, latent, mapping = ref.MiniMaxH3CharacterRefModR2V().encode(Clip(), character, text, 480, 864, length)
                    assert len(cond[0][1]["minimax_refs"]) == len(names)
                    assert len(mapping.splitlines()) == len(names)
                    assert latent["samples"].is_nested
                    assert all(mod.latent.device.type == "cpu" for mod, _ in mods)
                saved = ref.MiniMaxH3SaveCharacterRefMod().save(mods)["result"][0]
                loaded = bundle.load_bundle(str(Path(saved).with_suffix("")))
                assert len(loaded) == len(names)
                assert all(torch.equal(a.latent, b.latent) for (a, _), (b, _) in zip(mods, loaded))
            try:
                ref.MiniMaxH3CharacterRefModR2V().encode(Clip(), character, "<Picture 2>", 480, 864, 124)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing picture label was accepted")
            try:
                ref.MiniMaxH3CharacterRefModR2V().encode(Clip(), character, "x " * 6000, 480, 864, 124)
            except ValueError as exc:
                assert "4096" in str(exc)
            else:
                raise AssertionError("Oversized real-tokenizer prompt was accepted")
    print("RefMod CPU smoke passed: real native H3 + tokenizer, 1/2/1 images, full prompt, bundle roundtrip; model weights mocked, no GPU quality claim")


if __name__ == "__main__":
    main()
