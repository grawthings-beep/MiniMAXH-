"""Local prompt formatting and bounded appearance references; no model/API calls."""

from __future__ import annotations

import math


DEFAULT_DIRECTION = (
    "One continuous shot, fixed camera inside a dim room. "
    "Opening third: a barely open door hides the character; only a sliver of hair "
    "and one eye can be seen through the gap. "
    "Middle third: the character withdraws from the gap, leaving it empty for a "
    "brief beat, then peeks back in and places one hand on the door edge. "
    "Final third: the character opens the door and leans toward the camera. "
    "Their face fills the frame and changes into a playful smile. Hold the "
    "expression to finish. The camera does not move. No cuts or captions."
)
IDENTITY_NOTES = (
    "Preserve the reference's face shape, eye shape and size, hairstyle, "
    "outfit details, proportions, palette, and illustration style. "
    "Use a new pose and environment, not the reference composition."
)


def aligned_frames(seconds: float) -> int:
    if not math.isfinite(seconds) or not 5.0 <= seconds <= 15.0:
        raise ValueError("Choose 5–15 seconds; start with 5 for a VRAM/identity check.")
    frames = round(seconds * 24)
    return frames + (5 - frames) % 17


class MiniMaxH3CharacterPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.5}),
            "prompt_mode": (["direction", "full_prompt"],),
            "identity_notes": ("STRING", {"default": IDENTITY_NOTES, "multiline": True}),
            "direction": ("STRING", {"default": DEFAULT_DIRECTION, "multiline": True}),
            "soundscape": ("STRING", {"default": "Quiet room ambience, soft footsteps and a door opening. No dialogue.", "multiline": True}),
            "music": ("STRING", {"default": "N/A"}),
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "length")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = (
        "Formats ONE character reference as <Subject 1> from <Picture 1>, not a "
        "first frame. No LLM, translation, API, or automatic story writing. "
        "Use English visual directions; preserve dialogue language. full_prompt "
        "passes direction through unchanged and ignores the other text fields. "
        "Timing/identity requests are not hard constraints."
    )

    def build(self, duration_seconds, prompt_mode, identity_notes, direction, soundscape, music):
        length = aligned_frames(float(duration_seconds))
        if not direction.strip():
            raise ValueError("Write the opening, action and final reveal in direction.")
        if prompt_mode == "full_prompt":
            return direction, length
        if prompt_mode != "direction":
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> is the single character in <Picture 1>. "
            f"{identity_notes.strip()}\n\n"
            "summary:\n"
            f"[reference generation] A {length / 24:.2f}-second scene starring <Subject 1>. "
            "Use the image as an appearance reference, not as the first frame.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - retain character identity, costume and art style "
            "throughout; composition, pose and background follow the direction below.\n\n"
            "detailed_description:\n"
            f"{direction.strip()}\n\n"
            "overall_soundscape:\n"
            f"{soundscape.strip() or 'N/A'}\n\n"
            "non_diegetic_music:\n"
            f"{music.strip() or 'N/A'}"
        )
        return prompt, length


def reference_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    if min(width, height) < 32:
        raise ValueError("Character reference must be at least 32px on both axes.")
    if not 256 <= max_dimension <= 1536:
        raise ValueError("Reference long-edge cap must be between 256 and 1536px.")
    scale = min(1.0, max_dimension / max(width, height))
    # Match H3's 32px reference grid before encoding, without cropping or upscaling.
    return tuple(max(32, int(n * scale) // 32 * 32) for n in (width, height))


class MiniMaxH3CharacterReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "max_dimension": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = (
        "One appearance-reference image. Downscale only; retain the whole image "
        "and approximately its aspect ratio on H3's 32px grid. Paired with native "
        "R2V ref_image_size=max, this bounds reference tokens independently of "
        "output resolution. 1024 is the starting cap; try 768 if memory is tight."
    )

    def prepare(self, image, max_dimension):
        if len(image.shape) != 4 or image.shape[0] != 1:
            raise ValueError("Supply exactly one character image, not an image/video batch.")
        import comfy.utils

        width, height = reference_dimensions(image.shape[2], image.shape[1], max_dimension)
        rgb = image[..., :3]
        if (width, height) == (image.shape[2], image.shape[1]):
            return (rgb,)
        result = comfy.utils.common_upscale(
            rgb.movedim(-1, 1), width, height, "lanczos", "disabled"
        )
        return (result.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CharacterPrompt": MiniMaxH3CharacterPrompt,
    "MiniMaxH3CharacterReference": MiniMaxH3CharacterReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CharacterPrompt": "Character Direction / キャラ演出プロンプト",
    "MiniMaxH3CharacterReference": "Character Reference Budget / 参照画像サイズ上限",
}
