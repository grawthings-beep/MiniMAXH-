#!/usr/bin/env python3
"""CPU numerical integration check against the image's actual ComfyUI and KJNodes."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.comfyui_root), str(args.comfyui_root / "custom_nodes")]
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from comfy.ldm.minimax.model import Attention, MLP, DiTBlock, rope_rotation_table
    from comfy.model_patcher import ModelPatcher
    from minimax_h3_memory import NODE_CLASS_MAPPINGS
    from minimax_h3_memory.minimax_nodes import (
        MiniMaxFFNChunkPatch, minimax_attn_lowmem_forward,
    )

    assert set(NODE_CLASS_MAPPINGS) == {"MiniMaxChunkFeedForward", "MiniMaxLowVRAMAttention"}
    torch.manual_seed(713)
    # Repeated and uneven sequence/head splits catch shape and stale-state errors.
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            attn = Attention(40, 5, 8, 1e-6, dtype=dtype, operations=torch.nn)
            mlp = MLP(40, 64, dtype=dtype, operations=torch.nn)
            for length in (17, 9, 17):
                x = torch.randn(length, 40, dtype=dtype)
                expected = attn(x)
                for chunks in (1, 2, 4):
                    actual = minimax_attn_lowmem_forward(
                        attn, [x], transformer_options={"minimax_head_chunks": chunks})
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
                    assert actual.shape == x.shape and actual.dtype == dtype
                expected = mlp(x)
                actual = MiniMaxFFNChunkPatch(2, 8).__get__(mlp)(x)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
                assert actual.dtype == dtype

            # Exercise the actual node -> ModelPatcher -> block path, including
            # fused RMSNorm/RoPE and unpatch/repatch between two generations.
            block = DiTBlock(40, 5, 8, 64, 16, 1e-6, 1e-6,
                             dtype=dtype, operations=torch.nn)
            model = torch.nn.Module()
            model.diffusion_model = torch.nn.Module()
            model.diffusion_model.blocks = torch.nn.ModuleList([block])
            base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
            original_forward = block.forward
            patched = NODE_CLASS_MAPPINGS["MiniMaxLowVRAMAttention"].execute(base, 2).result[0]
            patched = NODE_CLASS_MAPPINGS["MiniMaxChunkFeedForward"].execute(patched, 2, 8).result[0]
            assert not base.object_patches and block.forward == original_forward
            for length in (17, 9):
                x = torch.randn(length, 40, dtype=dtype)
                t = torch.randn(1, 16, dtype=dtype)
                angles = torch.randn(length, 2).repeat(1, 2)
                rope = rope_rotation_table(angles, dtype)
                segments = [(0, 3, 0), (3, 6, 1), (6, length, 2)]
                expected = block(x.clone(), t, segments, rope)
                patched.patch_model(load_weights=False)
                try:
                    actual = block(x.clone(), t, segments, rope,
                                   transformer_options=patched.model_options["transformer_options"])
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                finally:
                    patched.unpatch_model(unpatch_weights=False)
                assert block.forward == original_forward
                torch.testing.assert_close(block(x.clone(), t, segments, rope), expected)
    print("H3 runtime: repeated FP32/BF16 attention, MLP, RoPE and ModelPatcher lifecycle passed on CPU")


if __name__ == "__main__":
    main()
