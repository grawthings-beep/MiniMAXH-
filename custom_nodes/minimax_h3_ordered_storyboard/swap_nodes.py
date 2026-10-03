"""Bounded, timestamp-resampled video references for H3 character replacement.

Decode only the requested clip on CPU; never call VIDEO.get_components() on a
whole uploaded movie. Original audio is an export choice, not a model input.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

FPS = 24
MAX_FRAMES = 362
SWAP_PROMPT = """Swap the main person in <Video 1> with the character in <Picture 1>. Use the replacement character's face, hair, outfit and art style. Match the source person's position, scale, pose and movement. Preserve the camera, background, lighting and objects. Keep the replacement consistent when leaving and re-entering the frame.

Soundscape: natural sounds matching the visible actions. No added dialogue.
Non-diegetic music: N/A.
"""


def validate_prompt(prompt, count):
    from .refmod_nodes import clean_full_prompt
    prompt = clean_full_prompt(prompt)
    for kind, number in re.findall(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", prompt, re.I):
        valid = (kind.lower() == "picture" and 1 <= int(number) <= count
                 or kind.lower() == "video" and int(number) == 1)
        if not valid:
            raise ValueError(f"Missing <{kind} {number}>: use <Video 1> and <Picture 1>–<Picture {count}>. Source audio is export-only.")
    return prompt


def canvas(width, height, megapixels, rotation=0, sar=1.0):
    if not math.isfinite(megapixels) or not .05 <= megapixels <= .8:
        raise ValueError("Character swap: choose 0.05–0.8MP; start at 0.4MP.")
    if min(width, height) < 32 or width * height > 40_000_000 or not 0.1 <= sar <= 10:
        raise ValueError("Unsupported source dimensions / sample aspect ratio.")
    w, h = width * sar, height
    if rotation % 2:
        w, h = h, w
    scale = min(1., math.sqrt(megapixels * 1_000_000 / (w * h)), 2048 / max(w, h))
    return max(32, int(w * scale) // 32 * 32), max(32, int(h * scale) // 32 * 32)


def read_clip(path, start, seconds, megapixels):
    """Return <=362 resized CPU frames, respecting PTS rather than assuming CFR."""
    import av
    import numpy as np
    import torch

    if not all(math.isfinite(v) for v in (start, seconds)) or start < 0 or not .92 <= seconds <= 15.1:
        raise ValueError("Start must be >=0; duration must be 0.92–15.1 seconds. Start with a short 4–5s shot.")
    limit = min(MAX_FRAMES, int(seconds * FPS + 1e-6))
    sampled = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Uploaded file has no video stream.")
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        declared_end = float(stream.duration * stream.time_base) if stream.duration else None
        if declared_end is not None and start >= declared_end:
            raise ValueError("Start time is beyond the end of the source video.")
        container.seek(int((origin + start) / stream.time_base), stream=stream, backward=True)
        previous, cached, plan = None, None, None
        last_end = 0.
        # Sample-and-hold uses each frame for its actual display interval. The
        # EOF interval is limited by duration metadata; no arbitrary tail freeze.
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video has no presentation timestamps; re-encode it with timestamps first.")
            timestamp = float(frame.pts * frame.time_base) - origin
            rotation = int(round(float(frame.rotation or 0) / 90)) % 4
            if plan is None:
                w, h = canvas(frame.width, frame.height, megapixels, rotation,
                              float(stream.sample_aspect_ratio or 1))
                plan = (w, h, rotation)
            if rotation != plan[2]:
                raise ValueError("Rotation changes within the clip; normalize the source first.")
            while len(sampled) < limit and start + len(sampled) / FPS < timestamp - 1e-7:
                if previous is None:
                    raise ValueError("Source starts after the requested timestamp; choose a later start.")
                if cached is None:
                    rw, rh = (plan[1], plan[0]) if plan[2] % 2 else plan[:2]
                    cached = previous.reformat(width=rw, height=rh, format="rgb24").to_ndarray()
                    if plan[2]:
                        cached = np.rot90(cached, k=plan[2]).copy()
                sampled.append(cached)
            if len(sampled) == limit:
                break
            previous, cached = frame, None
            duration = float(frame.duration * frame.time_base) if frame.duration else 1 / float(stream.average_rate or FPS)
            last_end = timestamp + duration
        else:
            end = min(last_end, declared_end) if declared_end is not None else last_end
            remaining = min(limit, max(0, int((end - start) * FPS + 1e-5)))
            if previous is not None and remaining > len(sampled):
                rw, rh = (plan[1], plan[0]) if plan[2] % 2 else plan[:2]
                cached = previous.reformat(width=rw, height=rh, format="rgb24").to_ndarray()
                if plan[2]:
                    cached = np.rot90(cached, k=plan[2]).copy()
                sampled.extend([cached] * (remaining - len(sampled)))
        # Only complete 24fps output intervals count, even when EOF falls inside
        # an interval. Remove incomplete tails before flooring to H3's grid.
        if declared_end is not None:
            sampled = sampled[:max(0, int((declared_end - start) * FPS + 1e-5))]
        count = len(sampled) - (len(sampled) - 5) % 17
        if count < 22:
            raise ValueError("Selected clip is too short: need at least 22 frames (0.917s). Choose an earlier start / longer source.")
        frames = torch.from_numpy(np.stack(sampled[:count])).to(dtype=torch.float32).div_(255.)
    del sampled, previous, cached
    audio, has_audio = read_audio(path, origin + start, count / FPS)
    return {"frames": frames, "audio": audio, "has_audio": has_audio, "length": count,
            "width": frames.shape[2], "height": frames.shape[1], "start": start,
            "duration": count / FPS}


def read_audio(path, absolute_start, seconds):
    """Crop/resample audio into a fixed-size CPU buffer, preserving stream offsets."""
    import av
    import numpy as np
    import torch
    rate = 48000
    waveform = np.zeros((2, round(seconds * rate)), dtype=np.float32)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate}, False
        stream = container.streams.audio[0]
        # Preroll gives the resampler/filter and compressed audio a stable start.
        container.seek(int(max(0, absolute_start - 1) / stream.time_base), stream=stream, backward=True)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)

        def put(frame):
            if frame.pts is None:
                raise ValueError("Audio has no timestamps; re-encode or remove the audio before uploading.")
            offset = round((float(frame.pts * frame.time_base) - absolute_start) * rate)
            samples = frame.to_ndarray()
            left, right = max(0, offset), min(waveform.shape[1], offset + samples.shape[1])
            if right > left:
                waveform[:, left:right] = samples[:, left-offset:right-offset]

        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts * frame.time_base) > absolute_start + seconds + .1:
                break
            for converted in resampler.resample(frame):
                put(converted)
        for converted in resampler.resample(None):
            put(converted)
    return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate}, True


class MiniMaxH3SwapClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",),
                             "start_seconds": ("FLOAT", {"default": 0., "min": 0., "max": 86400., "step": .1}),
                             "duration_seconds": ("FLOAT", {"default": 5.2, "min": .92, "max": 15.1, "step": .1}),
                             "megapixels": ("FLOAT", {"default": .4, "min": .05, "max": .8, "step": .05})}}
    RETURN_TYPES = ("H3_SWAP_CLIP", "STRING")
    RETURN_NAMES = ("source", "actual_clip")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Character Swap"

    def prepare(self, video, start_seconds=0., duration_seconds=5.2, megapixels=.4):
        import folder_paths
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        if not isinstance(video, VideoFromFile) or video.get_active_trim_window() != (0, 0):
            raise ValueError("Connect Load Video directly; set trimming only in this node.")
        source = video.get_stream_source()
        if not isinstance(source, str):
            raise ValueError("Use an uploaded file via Load Video, not an in-memory video.")
        path = Path(source).resolve()
        if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()) or not path.is_file():
            raise ValueError("Upload the source video to the ComfyUI input directory.")
        clip = read_clip(path, start_seconds, duration_seconds, megapixels)
        info = (f"24 fps | {clip['length']} frames | {clip['duration']:.3f}s | "
                f"{clip['width']} x {clip['height']} (before 2x)\n"
                f"Source window: {clip['start']:.3f}s–{clip['start']+clip['duration']:.3f}s\n"
                f"Original audio: {'available' if clip['has_audio'] else 'absent (silence)'}\n"
                "Tail trimmed to H3 17n+5 grid. No source frame freeze-padding.")
        return clip, info


class MiniMaxH3SwapPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"full_prompt": ("STRING", {"default": SWAP_PROMPT, "multiline": True, "dynamicPrompts": False})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Character Swap"

    def build(self, full_prompt):
        from .refmod_nodes import clean_full_prompt
        return (clean_full_prompt(full_prompt),)


class MiniMaxH3CharacterSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP", {"lazy": True}), "character": ("H3_CHARACTER_IMAGES", {"lazy": True}),
                             "source": ("H3_SWAP_CLIP",), "vae": ("VAE", {"lazy": True}),
                             "prompt": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/Character Swap"

    def check_lazy_status(self, clip=None, character=None, vae=None, **kwargs):
        return [key for key, value in (("clip", clip), ("character", character), ("vae", vae)) if value is None]

    def encode(self, clip, character, source, vae, prompt):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        from .refmod_nodes import MAX_IMAGES, PromptBudgetClip
        images = character["images"]
        if not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError("Use 1–8 reference pictures of the same replacement character.")
        prompt = validate_prompt(prompt, len(images))
        frames = source["frames"]
        n, h, w, channels = frames.shape
        if frames.device.type != "cpu" or channels != 3 or n % 17 != 5 or not 22 <= n <= MAX_FRAMES or w % 32 or h % 32:
            raise ValueError("Invalid prepared clip; connect the bounded source preparation node.")
        if (n, h, w) != (source["length"], source["height"], source["width"]):
            raise ValueError("Clip metadata does not match its frames.")
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Use the H3 INT8 video reference VAE.")
        # Match the LoRA author's reviewed example: the standard ComfyUI Ref2VA
        # node encodes both image and video references via the official INT8 VAE.
        # Keep only bounded video decoding and variable-count image loading here.
        result = MiniMaxH3ReferenceToVideo.execute(
            PromptBudgetClip(clip), prompt, w, h, n, ref_image_size="match", vae=vae,
            ref_images={f"ref_image_{i}": image for i, image in enumerate(images)},
            ref_videos={"ref_video_0": frames}).result
        return result[0], result[1]


class MiniMaxH3SwapAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("H3_SWAP_CLIP",), "generated": ("AUDIO",),
                             "audio_mode": (["source", "generated", "mute"], {"default": "source"})}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Character Swap"

    def select(self, source, generated, audio_mode="source"):
        import torch
        if audio_mode not in {"source", "generated", "mute"}:
            raise ValueError("Audio mode must be source, generated or mute.")
        if audio_mode == "mute":
            return ({"waveform": torch.zeros_like(source["audio"]["waveform"]), "sample_rate": 48000},)
        audio = source["audio"] if audio_mode == "source" else generated
        rate = int(audio["sample_rate"])
        count = round(source["duration"] * rate)
        wave = audio["waveform"].detach().cpu()[..., :count]
        if wave.shape[-1] < count:
            wave = torch.nn.functional.pad(wave, (0, count-wave.shape[-1]))
        return ({"waveform": wave, "sample_rate": rate},)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MiniMaxH3SwapClip, MiniMaxH3SwapPrompt, MiniMaxH3CharacterSwap, MiniMaxH3SwapAudio)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SwapClip": "Source clip / 区間・24fps・解像度",
    "MiniMaxH3SwapPrompt": "Replacement prompt / 全文1欄",
    "MiniMaxH3CharacterSwap": "Video + Character → native Ref2VA",
    "MiniMaxH3SwapAudio": "Audio / 元動画・生成音・無音",
}
