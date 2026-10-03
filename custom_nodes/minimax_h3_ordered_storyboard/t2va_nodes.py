"""Text-only H3 conditioning. No images, RefMods, downloads or global patches."""
from __future__ import annotations
import re
import math

ASPECT_RATIOS = ("9:16", "16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "21:9")

DEFAULT_PROMPT = """integrated_multimodal_description:
A single continuous waist-up shot in a sunlit courtyard, stylized anime animation. One adult woman with shoulder-length dark hair wears a simple blue casual jacket. The camera stays completely fixed. She looks into the lens with a playful, expectant expression, briefly shifts her weight, then quickly moves out through the right edge. She disappears completely; only the unchanged courtyard remains visible for a short beat. The same woman suddenly leans back into view from the right, closer to the camera this time, her face and shoulders angled diagonally toward the viewer. Her hair follows the quick movement. She brakes her movement, meets the viewer's eyes, and settles into a delighted smile. Keep the exit and return brisk, with a short quiet gap between them and a clear pause after the return. No cuts or captions.

overall_soundscape:
Soft outdoor ambience underneath a quick footstep and clothing rustle on the exit. Another footstep and rustle on the return, followed by a brief natural chuckle as she settles. No spoken dialogue.

non_diegetic_music:
Quiet playful pizzicato strings, a brief musical pause during the empty frame, then a light accent on the return. Keep the movement sounds audible.
"""


def text_only_prompt(value):
    from .refmod_nodes import clean_full_prompt
    prompt = clean_full_prompt(value)
    if re.search(r"<\s*(?:Picture|Video|Audio)\s+\d+\s*>", prompt, re.I):
        raise ValueError("画像なしT2VAでは<Picture>/<Video>/<Audio>参照を使えません。人物・動作・音を文章で指定してください。")
    return prompt


class MiniMaxH3T2VAPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.5}),
            "full_prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True, "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "length")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Text to Video"
    DESCRIPTION = "画像なし。映像・音・BGMを全文1欄へ。尺は24fpsの17n+5へ切り上げ（5秒→124frames）。"

    def build(self, duration_seconds, full_prompt):
        from .refmod_nodes import MiniMaxH3FullPrompt
        prompt, frames = MiniMaxH3FullPrompt().build(duration_seconds, full_prompt)
        return text_only_prompt(prompt), frames


class MiniMaxH3T2VAResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "aspect_ratio": (list(ASPECT_RATIOS), {"default": "9:16"}),
            "megapixels": ("FLOAT", {"default": .4, "min": .1, "max": .8, "step": .1}),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "dimensions"
    CATEGORY = "MiniMax H3/Text to Video"
    DESCRIPTION = "06と同じ画素予算（1MP=1,000,000px）、32px単位で切り下げ。0.4MP・9:16は448x832。"

    def dimensions(self, aspect_ratio, megapixels):
        if aspect_ratio not in ASPECT_RATIOS or not math.isfinite(megapixels) or not .1 <= megapixels <= .8:
            raise ValueError("Choose a listed aspect ratio and 0.1–0.8 MP; start at 0.4 MP.")
        w, h = map(int, aspect_ratio.split(":"))
        scale = math.sqrt(megapixels * 1_000_000 / (w*h))
        return max(32, math.floor(w*scale/32)*32), max(32, math.floor(h*scale/32)*32)


class MiniMaxH3TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "prompt": ("STRING", {"forceInput": True}),
            "width": ("INT", {"default": 448, "min": 32, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 832, "min": 32, "max": 2048, "step": 32}),
            "length": ("INT", {"default": 124, "min": 124, "max": 362, "step": 17}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "LATENT", "actual_clip")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/Text to Video"
    DESCRIPTION = "Native H3 T2VA（画像・動画・RefModなし）。元動画は最大0.8MP。生成の成功やOOM回避は保証されません。"

    def encode(self, clip, prompt, width, height, length):
        from .refmod_nodes import PromptBudgetClip
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        if any(type(v) is not int for v in (width, height, length)):
            raise ValueError("T2VA dimensions and frame count must be integers.")
        if not all(32 <= v <= 2048 and v % 32 == 0 for v in (width, height)) or width * height > 800_000:
            raise ValueError("元動画は32px単位・0.8MP以下で生成してください。まず0.4MPを推奨します。")
        if not 124 <= length <= 362 or length % 17 != 5:
            raise ValueError("T2VA uses 124–362 frames on the 17n+5 grid at 24fps.")
        prompt = text_only_prompt(prompt)
        # The native FL2VA node implements T2VA when BOTH keyframes are absent.
        # Its VAE argument is only used for keyframe encoding: no VAE load here.
        result = MiniMaxH3ImageToVideo.execute(
            clip=PromptBudgetClip(clip), vae=None, prompt=prompt,
            width=width, height=height, length=length, first_frame=None, last_frame=None)
        return result[0], result[1], (
            f"T2VA | {width} x {height} | {length} frames / 24fps = {length/24:.3f}s | "
            "no reference inputs | native resolution, no upscale")


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3T2VAResolution": MiniMaxH3T2VAResolution,
    "MiniMaxH3T2VAPrompt": MiniMaxH3T2VAPrompt,
    "MiniMaxH3TextToVideo": MiniMaxH3TextToVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3T2VAResolution": "T2VA Resolution / 縦横比・画素予算",
    "MiniMaxH3T2VAPrompt": "T2VA Full Prompt / 画像なし・全文1欄",
    "MiniMaxH3TextToVideo": "MiniMax H3 Text to Video + Audio",
}
