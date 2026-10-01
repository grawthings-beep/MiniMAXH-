"""CPU contract test for the pinned decoder (synthetic tensors, no model download)."""
import argparse
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true", help="Stub device management for lightweight CPU CI")
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from comfy.nested_tensor import NestedTensor

    if args.isolated:
        # Exercise the real PyTorch packing/decoder logic without a CUDA runtime
        # or the full Comfy server's dependency set. Docker uses the real modules.
        import comfy
        management = ModuleType("comfy.model_management")
        management.intermediate_device = lambda: torch.device("cpu")
        management.load_models_gpu = lambda *a, **kw: None
        management.cuda_device_context = lambda device: contextlib.nullcontext()
        management.get_torch_device = lambda: torch.device("cpu")
        sys.modules[management.__name__] = management
        comfy.model_management = management
        utils = ModuleType("comfy.utils")
        sys.modules[utils.__name__] = utils
        comfy.utils = utils

    package = args.comfyui_root / "custom_nodes/minimax_h3_x2_vae"
    spec = importlib.util.spec_from_file_location("x2_runtime", package / "__init__.py",
                                                 submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decoder_module = sys.modules["x2_runtime.vae_decode"]
    node = module.NODE_CLASS_MAPPINGS["MiniMaxH3VAEDecodeFast"]()
    inputs = node.INPUT_TYPES()
    assert inputs["required"]["samples"] == ("LATENT",)
    assert inputs["required"]["vae"] == ("VAE",)
    assert node.RETURN_TYPES == ("IMAGE",)

    mean, std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    inner = SimpleNamespace(
        tiling=False, tile_size=512, tile_overlap_min=128, pixel_mean=mean, pixel_std=std,
        decoder=SimpleNamespace(proj_out=SimpleNamespace(weight=torch.zeros(48, 1)),
                                patch_size_t=1, patch_size=2, out_channels=3))
    inner.decode_output_shape = lambda shape: (shape[0], inner.decoder.out_channels, 5, 2, 2)

    def decode(samples, output_buffer):
        assert inner.tiling and inner.tile_size == 256 and inner.tile_overlap_min == 64
        assert inner.pixel_mean.shape == (1, 12, 1, 1, 1)
        output_buffer.copy_(torch.arange(12).reshape(1, 12, 1, 1, 1).expand_as(output_buffer))

    inner.decode = decode
    vae = SimpleNamespace(
        first_stage_model=inner, device=torch.device("cpu"), vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32, process_output=lambda pixels: None,
        throw_exception_if_invalid=lambda: None, memory_used_decode=lambda shape, dtype: 1,
        disable_offload=False, patcher=SimpleNamespace(is_dynamic=lambda: False))
    video, audio = torch.zeros(1, 24, 2, 1, 1), torch.ones(1, 32, 2, 4)
    samples = {"samples": NestedTensor([video, audio])}
    expected = torch.nn.functional.pixel_shuffle(
        torch.arange(12).float().reshape(1, 12, 1, 1).expand(5, 12, 2, 2), 2).movedim(1, -1)
    with patch.object(decoder_module.model_management, "load_models_gpu"), \
         patch.object(decoder_module.model_management, "cuda_device_context", lambda device: contextlib.nullcontext()):
        for _ in range(2):
            actual = node.decode(samples, vae)[0]
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (5, 4, 4, 3)
            assert inner.decoder.out_channels == 3 and inner.pixel_mean is mean and inner.pixel_std is std
            assert not inner.tiling and inner.tile_size == 512 and inner.tile_overlap_min == 128
            torch.testing.assert_close(audio, torch.ones_like(audio))
        inner.decode = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("synthetic decode failure"))
        try:
            node.decode(samples, vae)
        except RuntimeError as error:
            assert str(error) == "synthetic decode failure"
        else:
            raise AssertionError("Expected synthetic failure")
        assert inner.decoder.out_channels == 3 and inner.pixel_mean is mean and inner.pixel_std is std
        assert not inner.tiling and inner.tile_size == 512
    print("X2 decoder CPU contract: nested AV, PixelShuffle, repeat decode, error-state restoration passed; GPU quality untested")


if __name__ == "__main__":
    main()
