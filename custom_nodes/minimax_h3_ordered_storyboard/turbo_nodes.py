"""One-control LightX2V Turbo profile selection for MiniMax H3 FL2VA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TURBO_8STEP_MODEL = (
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
TURBO_4STEP_MODEL = (
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
)

PROFILE_8STEP = "8-step v1.0 768p (recommended)"
PROFILE_4STEP = "4-step v1.2 768p (fastest)"


@dataclass(frozen=True)
class TurboProfile:
    model_name: str
    steps: int


TURBO_PROFILES = {
    PROFILE_8STEP: TurboProfile(TURBO_8STEP_MODEL, 8),
    PROFILE_4STEP: TurboProfile(TURBO_4STEP_MODEL, 4),
}


class MiniMaxH3TurboProfile:
    """Apply the selected pinned Turbo LoRA and emit its matching step count."""

    def __init__(self):
        self._loaded_lora: tuple[str, Any, Any] | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "profile": (tuple(TURBO_PROFILES),),
            }
        }

    RETURN_TYPES = ("MODEL", "INT")
    RETURN_NAMES = ("MODEL", "steps")
    FUNCTION = "apply_profile"
    CATEGORY = "MiniMax H3/Turbo"
    DESCRIPTION = (
        "Selects a pinned LightX2V FL2VA 768p Turbo LoRA and sends its matching "
        "4- or 8-step count to BasicScheduler. Both profiles use video/audio "
        "sigma shifts 6/3, Euler, and the simple scheduler."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, profile, **_kwargs):
        if profile not in TURBO_PROFILES:
            return f"Unknown MiniMax H3 Turbo profile: {profile!r}"
        return True

    def apply_profile(self, model, profile):
        if profile not in TURBO_PROFILES:
            raise ValueError(f"Unknown MiniMax H3 Turbo profile: {profile!r}")

        import folder_paths
        from comfy import sd, utils

        selected = TURBO_PROFILES[profile]
        lora_path = folder_paths.get_full_path_or_raise("loras", selected.model_name)
        resolved_path = str(Path(lora_path).resolve())

        lora = None
        metadata = None
        if self._loaded_lora is not None and self._loaded_lora[0] == resolved_path:
            lora, metadata = self._loaded_lora[1:]
        if lora is None:
            lora, metadata = utils.load_torch_file(
                resolved_path,
                safe_load=True,
                return_metadata=True,
            )
            # Keep only the active profile resident. Switching modes releases the
            # previous 1.96 GB state dict instead of retaining both Turbo LoRAs.
            self._loaded_lora = (resolved_path, lora, metadata)

        patched_model, _ = sd.load_lora_for_models(
            model,
            None,
            lora,
            1.0,
            0.0,
            lora_metadata=metadata,
        )
        return patched_model, selected.steps


NODE_CLASS_MAPPINGS = {"MiniMaxH3TurboProfile": MiniMaxH3TurboProfile}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboProfile": "MiniMax H3 Turbo Profile (4/8-step 768p)"
}


__all__ = [
    "MiniMaxH3TurboProfile",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PROFILE_4STEP",
    "PROFILE_8STEP",
    "TURBO_4STEP_MODEL",
    "TURBO_8STEP_MODEL",
    "TURBO_PROFILES",
]
