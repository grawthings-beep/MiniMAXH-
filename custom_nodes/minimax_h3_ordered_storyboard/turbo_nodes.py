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
CREATOR_LORA_DISABLED = "Disabled (Turbo/base model only)"
DEFAULT_CREATOR_LORAS = (
    "HMNSFW_AIO_V2.safetensors",
    "hmmotion_minimax-h3_epoch12.safetensors",
)


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
            },
            "optional": {
                "profile_control": ("H3_TURBO_PROFILE",),
            },
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

    def apply_profile(self, model, profile, profile_control=None):
        if profile_control is not None:
            profile = profile_control
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


class MiniMaxH3TurboLoRAControl:
    """Visible top-level selector for the pinned Turbo LoRA profile."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profile": (tuple(TURBO_PROFILES),)}}

    RETURN_TYPES = ("H3_TURBO_PROFILE",)
    RETURN_NAMES = ("turbo_profile",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Controls"
    DESCRIPTION = (
        "Keep this node outside the H3 subgraph. It selects the 8-step quality "
        "or 4-step fastest 768p Turbo LoRA used inside the workflow."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, profile, **_kwargs):
        if profile not in TURBO_PROFILES:
            return f"Unknown MiniMax H3 Turbo profile: {profile!r}"
        return True

    def select(self, profile):
        if profile not in TURBO_PROFILES:
            raise ValueError(f"Unknown MiniMax H3 Turbo profile: {profile!r}")
        return (profile,)


def _creator_lora_choices() -> tuple[str, ...]:
    try:
        import folder_paths

        installed = folder_paths.get_filename_list("loras")
    except (ImportError, AttributeError):
        installed = []
    turbo_names = {profile.model_name for profile in TURBO_PROFILES.values()}
    choices = [name for name in installed if name not in turbo_names]
    if not choices:
        choices = list(DEFAULT_CREATOR_LORAS)
    for name in reversed(DEFAULT_CREATOR_LORAS):
        if name not in choices:
            continue
        choices.remove(name)
        choices.insert(0, name)
    return (CREATOR_LORA_DISABLED, *choices)


class MiniMaxH3CreatorLoRAControl:
    """Visible top-level selector for the optional creator/style LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_creator_lora_choices(),),
                "strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("H3_CREATOR_LORA",)
    RETURN_NAMES = ("creator_lora",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Controls"
    DESCRIPTION = (
        "Selects an optional creator/style LoRA outside the H3 subgraph. Strength "
        "0.0 or Disabled bypasses loading and patching completely."
    )

    def select(self, lora_name, strength):
        return ({"lora_name": str(lora_name), "strength": float(strength)},)


class MiniMaxH3CreatorLoRAApply:
    """Internal model-only LoRA applicator driven by the visible control node."""

    def __init__(self):
        self._loaded_lora: tuple[str, Any, Any] | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "creator_lora": ("H3_CREATOR_LORA",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Internal"
    DESCRIPTION = "Applies the creator LoRA selected by MiniMax H3 Creator LoRA Control."

    def apply(self, model, creator_lora):
        lora_name = str(creator_lora.get("lora_name", CREATOR_LORA_DISABLED))
        strength = float(creator_lora.get("strength", 0.0))
        if lora_name == CREATOR_LORA_DISABLED or strength == 0.0:
            # Do not even deserialize a 300 MB LoRA when the visible control is off.
            self._loaded_lora = None
            return (model,)

        import folder_paths
        from comfy import sd, utils

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
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
            self._loaded_lora = (resolved_path, lora, metadata)

        patched_model, _ = sd.load_lora_for_models(
            model,
            None,
            lora,
            strength,
            0.0,
            lora_metadata=metadata,
        )
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboProfile": MiniMaxH3TurboProfile,
    "MiniMaxH3TurboLoRAControl": MiniMaxH3TurboLoRAControl,
    "MiniMaxH3CreatorLoRAControl": MiniMaxH3CreatorLoRAControl,
    "MiniMaxH3CreatorLoRAApply": MiniMaxH3CreatorLoRAApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboProfile": "MiniMax H3 Turbo Profile (4/8-step 768p)",
    "MiniMaxH3TurboLoRAControl": "Turbo LoRA — 4/8-step 768p",
    "MiniMaxH3CreatorLoRAControl": "Optional Creator LoRA — Select / Strength",
    "MiniMaxH3CreatorLoRAApply": "Apply Visible Creator LoRA Control",
}


__all__ = [
    "MiniMaxH3TurboProfile",
    "MiniMaxH3TurboLoRAControl",
    "MiniMaxH3CreatorLoRAControl",
    "MiniMaxH3CreatorLoRAApply",
    "CREATOR_LORA_DISABLED",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PROFILE_4STEP",
    "PROFILE_8STEP",
    "TURBO_4STEP_MODEL",
    "TURBO_8STEP_MODEL",
    "TURBO_PROFILES",
]
