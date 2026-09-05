"""One-control LightX2V Turbo profile selection for MiniMax H3 FL2VA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TURBO_8STEP_MODEL = (
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
)
TURBO_4STEP_MODEL = (
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
)

PROFILE_8STEP = "8-step v1.0 FL2VA (recommended)"
PROFILE_4STEP = "4-step v1.2 768p (fastest)"
TURBO_DEFAULT_STRENGTH = 0.70
CREATOR_LORA_DISABLED = "Disabled (Turbo/base model only)"
ANIME_MOTION_MODEL = "H3_Motion_Booster_anime.safetensors"
ANIME_STYLE_MODEL = "NSFW_ANIME_V7_H3-step00019500.safetensors"
DEFAULT_CREATOR_LORAS = (
    ANIME_MOTION_MODEL,
    ANIME_STYLE_MODEL,
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
                "strength": (
                    "FLOAT",
                    {"default": TURBO_DEFAULT_STRENGTH, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
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
        "Selects a pinned LightX2V FL2VA Turbo LoRA and sends its matching "
        "4- or 8-step count to BasicScheduler. Turbo strength is independently "
        "adjustable; both profiles use video/audio "
        "sigma shifts 6/3, Euler, and the simple scheduler."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, profile, **_kwargs):
        if profile not in TURBO_PROFILES:
            return f"Unknown MiniMax H3 Turbo profile: {profile!r}"
        return True

    def apply_profile(
        self, model, profile, strength=TURBO_DEFAULT_STRENGTH, profile_control=None
    ):
        if profile_control is not None:
            if isinstance(profile_control, dict):
                profile = str(profile_control.get("profile", profile))
                strength = float(profile_control.get("strength", strength))
            else:  # backwards compatibility with the original string control
                profile = profile_control
        if profile not in TURBO_PROFILES:
            raise ValueError(f"Unknown MiniMax H3 Turbo profile: {profile!r}")
        strength = float(strength)
        if not 0.0 <= strength <= 1.5:
            raise ValueError("MiniMax H3 Turbo strength must be between 0.0 and 1.5")

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
            strength,
            0.0,
            lora_metadata=metadata,
        )
        return patched_model, selected.steps


class MiniMaxH3TurboLoRAControl:
    """Visible top-level selector for the pinned Turbo LoRA profile."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": (tuple(TURBO_PROFILES),),
                "strength": (
                    "FLOAT",
                    {"default": TURBO_DEFAULT_STRENGTH, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("H3_TURBO_PROFILE",)
    RETURN_NAMES = ("turbo_profile",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Controls"
    DESCRIPTION = (
        "Keep this node outside the H3 subgraph. It selects the 8-step quality "
        "or 4-step fastest Turbo LoRA and its strength used inside the workflow."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, profile, **_kwargs):
        if profile not in TURBO_PROFILES:
            return f"Unknown MiniMax H3 Turbo profile: {profile!r}"
        return True

    def select(self, profile, strength=TURBO_DEFAULT_STRENGTH):
        if profile not in TURBO_PROFILES:
            raise ValueError(f"Unknown MiniMax H3 Turbo profile: {profile!r}")
        return ({"profile": profile, "strength": float(strength)},)


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
    """Visible top-level two-slot selector for optional creator/style LoRAs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_1_name": (_creator_lora_choices(),),
                "lora_1_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
                "lora_2_name": (_creator_lora_choices(),),
                "lora_2_strength": (
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
        "Stacks up to two optional creator/style LoRAs outside the H3 subgraph. "
        "Each slot independently bypasses loading when strength is 0.0 or Disabled."
    )

    def select(
        self,
        lora_1_name,
        lora_1_strength,
        lora_2_name=CREATOR_LORA_DISABLED,
        lora_2_strength=0.0,
    ):
        return (
            {
                "loras": [
                    {
                        "lora_name": str(lora_1_name),
                        "strength": float(lora_1_strength),
                    },
                    {
                        "lora_name": str(lora_2_name),
                        "strength": float(lora_2_strength),
                    },
                ]
            },
        )


class MiniMaxH3CreatorLoRAApply:
    """Internal model-only LoRA applicator driven by the visible control node."""

    def __init__(self):
        self._loaded_loras: dict[str, tuple[Any, Any]] = {}

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
        configurations = creator_lora.get("loras")
        if not isinstance(configurations, list):
            configurations = [creator_lora]  # backwards-compatible one-slot payload
        active = [
            (str(item.get("lora_name", CREATOR_LORA_DISABLED)), float(item.get("strength", 0.0)))
            for item in configurations
            if isinstance(item, dict)
            and str(item.get("lora_name", CREATOR_LORA_DISABLED)) != CREATOR_LORA_DISABLED
            and float(item.get("strength", 0.0)) != 0.0
        ]
        names = [name for name, _ in active]
        if len(names) != len(set(names)):
            raise ValueError("Select each creator LoRA in at most one slot")
        if not active:
            # Do not deserialize creator LoRAs when both visible slots are off.
            self._loaded_loras.clear()
            return (model,)

        import folder_paths
        from comfy import sd, utils

        next_cache: dict[str, tuple[Any, Any]] = {}
        patched_model = model
        for lora_name, strength in active:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            resolved_path = str(Path(lora_path).resolve())
            cached = self._loaded_loras.get(resolved_path)
            if cached is None:
                cached = utils.load_torch_file(
                    resolved_path,
                    safe_load=True,
                    return_metadata=True,
                )
            lora, metadata = cached
            next_cache[resolved_path] = cached
            patched_model, _ = sd.load_lora_for_models(
                patched_model,
                None,
                lora,
                strength,
                0.0,
                lora_metadata=metadata,
            )
        self._loaded_loras = next_cache
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboProfile": MiniMaxH3TurboProfile,
    "MiniMaxH3TurboLoRAControl": MiniMaxH3TurboLoRAControl,
    "MiniMaxH3CreatorLoRAControl": MiniMaxH3CreatorLoRAControl,
    "MiniMaxH3CreatorLoRAApply": MiniMaxH3CreatorLoRAApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboProfile": "MiniMax H3 Turbo Profile (4/8-step)",
    "MiniMaxH3TurboLoRAControl": "Turbo LoRA — Profile / Strength",
    "MiniMaxH3CreatorLoRAControl": "Optional Creator LoRA Stack — 2 Slots",
    "MiniMaxH3CreatorLoRAApply": "Apply Visible Creator LoRA Control",
}


__all__ = [
    "MiniMaxH3TurboProfile",
    "MiniMaxH3TurboLoRAControl",
    "MiniMaxH3CreatorLoRAControl",
    "MiniMaxH3CreatorLoRAApply",
    "CREATOR_LORA_DISABLED",
    "ANIME_MOTION_MODEL",
    "ANIME_STYLE_MODEL",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PROFILE_4STEP",
    "PROFILE_8STEP",
    "TURBO_4STEP_MODEL",
    "TURBO_8STEP_MODEL",
    "TURBO_DEFAULT_STRENGTH",
    "TURBO_PROFILES",
]
