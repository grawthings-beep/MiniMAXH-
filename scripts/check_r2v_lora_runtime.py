#!/usr/bin/env python3
"""Real ComfyUI LoRA loader/patcher CPU smoke with tiny synthetic weights (not GPU quality)."""
import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_nodes"))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from safetensors.torch import save_file
    import folder_paths
    import comfy.lora
    from comfy.model_base import MiniMaxH3
    from comfy.model_patcher import ModelPatcher
    import minimax_h3_ordered_storyboard as package

    cls = package.NODE_CLASS_MAPPINGS["MiniMaxH3R2VLoRA"]
    node = cls()
    model = object.__new__(MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(unet_config={})
    model.diffusion_model = torch.nn.Module()
    model.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.diffusion_model.blocks[0].linear = torch.nn.Linear(4, 4, bias=False)
    base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    key = "diffusion_model.blocks.0.linear.weight"
    with tempfile.TemporaryDirectory(prefix="h3-lora-cpu-") as temp:
        folder_paths.add_model_folder_path("loras", temp)
        name = "r2v-smoke.safetensors"
        save_file({"blocks.0.linear.lora_A.weight": torch.ones(1, 4),
                   "blocks.0.linear.lora_B.weight": torch.ones(4, 1)},
                  str(Path(temp) / name), metadata={"fixture": "synthetic CPU smoke"})
        assert name in cls.INPUT_TYPES()["required"]["lora_name"][0]
        for strength in (0.4, 0.2, 0.4):
            patched = node.apply_lora(base, name, strength, True)[0]
            assert patched is not base and not base.patches
            assert len(patched.patches[key]) == 1  # No accumulation across runs.
            result = comfy.lora.calculate_weight(patched.patches[key], torch.zeros(4, 4), key)
            assert torch.allclose(result, torch.full((4, 4), strength))
            assert patched.get_attachment("lora_metadata")["fixture"] == "synthetic CPU smoke"
        for enabled, strength in ((False, 0.4), (True, 0.0)):
            assert cls.VALIDATE_INPUTS("not-installed.safetensors", strength, enabled) is True
            assert node.apply_lora(base, "not-installed.safetensors", strength, enabled)[0] is base
            assert node._loaded_lora is None and not base.patches
    print("R2VA LoRA CPU smoke passed: registration, native H3 key mapping, safetensors metadata, strength, no stacking, OFF/zero bypass; synthetic weights only")


if __name__ == "__main__":
    main()
