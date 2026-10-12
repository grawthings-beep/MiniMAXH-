"""Generate an H3 soundtrack against a pinned, uploaded video.

Only the audio half of the joint AV latent is denoised. The delivered MP4
remuxes the original encoded video stream; VAE-decoded frames are never saved.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave


DEFAULT_AUDIO_PROMPT = """Use <Video 1> only to understand the visible actions and their timing. Generate a restrained, synchronized soundtrack for the fixed video: natural ambience appropriate to its setting, small contact sounds and clothing movement only when visibly caused, and one soft nonverbal breath or brief breath-laugh only if the person's expression supports it. Leave quiet moments quiet. No spoken words, narration, singing, exaggerated cartoon effects, or music."""


def aligned_frames(duration: float) -> int:
    if not math.isfinite(duration) or not 4.5 <= duration <= 15.0:
        raise ValueError("Use one 4.5–15 second clip. Trim longer videos before uploading.")
    # MP4 time bases commonly round a 124/24-second H3 clip to 5.167s.
    # Ignore up to a quarter-frame of container rounding, but not a real frame.
    count = max(124, math.ceil(duration * 24 - .25))
    while count % 17 != 5:
        count += 1
    if count > 362:
        raise ValueError("The source clip is too long for the 15-second H3 grid.")
    return count


def source_path(video):
    import folder_paths
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    if not isinstance(video, VideoFromFile) or video.get_active_trim_window() != (0, 0):
        raise ValueError("Connect Load Video directly. Trim the source file before uploading.")
    raw = video.get_stream_source()
    if not isinstance(raw, str):
        raise ValueError("Upload a video file via Load Video, not an in-memory video.")
    path = Path(raw).resolve()
    if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()) or not path.is_file():
        raise ValueError("The source video must be a file in ComfyUI/input.")
    return path


class MiniMaxH3VideoAudioClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",),
                             "analysis_megapixels": ("FLOAT", {"default": .2, "min": .1, "max": .4, "step": .05})}}

    RETURN_TYPES = ("H3_AUDIO_CLIP", "STRING")
    RETURN_NAMES = ("source", "actual_clip")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Video Audio"
    DESCRIPTION = "4.5–15秒の動画を24fps/17n+5に整列。解析だけ0.2MP、出力は元の映像ストリームをコピー。"

    def prepare(self, video, analysis_megapixels=.2):
        import torch
        from .swap_nodes import read_clip

        if not math.isfinite(analysis_megapixels) or not .1 <= analysis_megapixels <= .4:
            raise ValueError("Analysis resolution must be 0.1–0.4 MP.")
        path = source_path(video)
        import av
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("The uploaded file has no video stream.")
            stream = container.streams.video[0]
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
            elif stream.frames and stream.average_rate:
                duration = float(stream.frames / stream.average_rate)
            else:
                duration = float(video.get_duration())
        target = aligned_frames(duration)
        clip = read_clip(path, 0., min(duration, 15.), analysis_megapixels, include_audio=False)
        frames = clip["frames"]
        if frames.shape[0] > target:
            frames = frames[:target]
        elif frames.shape[0] < target:
            frames = torch.cat((frames, frames[-1:].expand(target - frames.shape[0], -1, -1, -1)), dim=0)
        clip.update(frames=frames, length=target, duration=duration, path=str(path),
                    width=frames.shape[2], height=frames.shape[1])
        return clip, (f"Original: {duration:.3f}s | H3 analysis: {target} frames / 24fps | "
                      f"{clip['width']} x {clip['height']} | video stream copied unchanged at export")


class MiniMaxH3VideoAudioPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_direction": ("STRING", {"default": DEFAULT_AUDIO_PROMPT,
                                                            "multiline": True, "dynamicPrompts": False})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Video Audio"

    def build(self, audio_direction):
        from .refmod_nodes import clean_full_prompt
        prompt = clean_full_prompt(audio_direction)
        tags = re.findall(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", prompt, re.I)
        if any(kind.lower() != "video" or number != "1" for kind, number in tags):
            raise ValueError("Only <Video 1> is connected in this audio-only workflow.")
        if not any(kind.lower() == "video" for kind, _ in tags):
            prompt = "Use <Video 1> as the fixed visual source. " + prompt
        return (prompt,)


class MiniMaxH3VideoAudioCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",), "source": ("H3_AUDIO_CLIP",),
                             "prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/Video Audio"
    DESCRIPTION = "元映像をH3映像latentへ固定。Qwenへ2fpsの動画参照を渡し、音声latentだけをdenoise。"

    def encode(self, clip, vae, source, prompt):
        import torch
        import comfy.model_management
        import comfy.nested_tensor
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        from .refmod_nodes import PromptBudgetClip

        frames = source["frames"]
        n, h, w, channels = frames.shape
        if (frames.device.type != "cpu" or channels != 3 or n != source["length"] or
                n % 17 != 5 or not 124 <= n <= 362 or w % 32 or h % 32 or w*h > 400_000):
            raise ValueError("Invalid bounded video clip. Reconnect Video Audio Clip.")
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Connect the standard H3 INT8 video VAE, not X2 Detail or audio VAE.")
        sample_indices = list(range(0, n, 12))
        refs = [{"type": "video", "data": frames[sample_indices],
                 "timestamps": [i / 24. for i in sample_indices]}]
        budget_clip = PromptBudgetClip(clip)
        tokens = budget_clip.tokenize(prompt, minimax_ref_items=refs)
        positive = budget_clip.encode_from_tokens_scheduled(tokens)
        latent, _ = _empty_av_latent(w, h, n)
        _, empty_audio = latent["samples"].unbind()
        with torch.inference_mode():
            encoded_video = vae.encode(frames)
        expected = (1, 24, latent["samples"].unbind()[0].shape[2], h//16, w//16)
        if tuple(encoded_video.shape) != expected:
            raise ValueError(f"H3 video VAE shape mismatch: {tuple(encoded_video.shape)} != {expected}")
        encoded_video = encoded_video.detach().to(comfy.model_management.intermediate_device())
        latent["samples"] = comfy.nested_tensor.NestedTensor((encoded_video, empty_audio))
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.zeros_like(encoded_video), torch.ones_like(empty_audio)))
        return positive, latent


def write_wave(path: Path, audio: dict, duration: float):
    """Create a bounded stereo WAV; pad or trim to the original video duration."""
    import numpy as np
    import torch

    rate = int(audio["sample_rate"])
    waveform = audio["waveform"]
    if rate < 8000 or rate > 192000 or waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("H3 generated audio has an unsupported shape or sample rate.")
    wanted = round(duration * rate)
    sound = waveform.detach().float().cpu()[0, :, :wanted]
    if sound.shape[1] < wanted:
        sound = torch.nn.functional.pad(sound, (0, wanted - sound.shape[1]))
    if sound.shape[0] == 1:
        sound = sound.expand(2, -1)
    pcm = (sound.clamp(-1, 1).movedim(0, 1).numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


class MiniMaxH3VideoAudioSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("H3_AUDIO_CLIP",), "generated": ("AUDIO",),
                             "filename_prefix": ("STRING", {"default": "video/MiniMax_H3_08_Video_Audio"})}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "saved_path")
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/Video Audio"
    OUTPUT_NODE = True
    DESCRIPTION = "元MP4映像をstream-copyし、H3生成音声へ差し替える。映像を再エンコードしません。"

    def save(self, source, generated, filename_prefix="video/MiniMax_H3_08_Video_Audio"):
        import folder_paths
        from comfy_api.latest import InputImpl, io, ui

        path = Path(source["path"]).resolve()
        if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()) or not path.is_file():
            raise ValueError("Source video is no longer an uploaded ComfyUI input file.")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to preserve the original video stream.")
        output_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), source["width"], source["height"])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f"{filename}_{counter:05}_.mp4"
        destination = Path(output_dir) / output_file
        partial = destination.with_name(destination.stem + ".partial.mp4")
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="h3-video-audio-", dir=temp_dir) as temporary:
                wav = Path(temporary) / "generated.wav"
                write_wave(wav, generated, float(source["duration"]))
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-i", str(path), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                           "-movflags", "+faststart", str(partial)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
                if result.returncode:
                    raise RuntimeError(f"Video/audio mux failed: {result.stderr[-2000:]}")
                os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        video = InputImpl.VideoFromFile(str(destination))
        preview = ui.PreviewVideo([ui.SavedResult(output_file, subfolder, io.FolderType.output)])
        return io.NodeOutput(video, str(destination), ui=preview)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MiniMaxH3VideoAudioClip, MiniMaxH3VideoAudioPrompt,
    MiniMaxH3VideoAudioCondition, MiniMaxH3VideoAudioSave)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3VideoAudioClip": "Video Audio / 元動画を解析",
    "MiniMaxH3VideoAudioPrompt": "Video Audio / 効果音・吐息の指示",
    "MiniMaxH3VideoAudioCondition": "Video locked / H3 audio-only sampling",
    "MiniMaxH3VideoAudioSave": "Original MP4 + H3 audio / 映像維持",
}
