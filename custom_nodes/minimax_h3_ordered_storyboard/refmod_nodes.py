"""Single full prompt + bounded, independent image RefMods for native H3 R2VA.

Uses the pinned upstream RefMod core/bundle format, not a LoRA or face restorer.
No model downloads, network calls, global patches, or GPU tensor caches here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import types

REFMOD_COMMIT = "f9462081e28794389b5a6c5067eb327412ad8ee7"
MAX_IMAGES = 8
MAX_PROMPT_CHARS = 16000
DEFAULT_FULL_PROMPT = """subject_definitions:
<Subject 1> is the same character shown in all supplied reference pictures. Preserve the face shape, eye shape and size, hairstyle, outfit and rendering style. These are appearance references, not the opening frame.

summary:
[reference generation] A short playful scene with <Subject 1>.

retention_analysis:
<Subject 1>: fully_preserved - character identity and outfit. Use the composition, pose and background described below.

detailed_description:
One continuous medium close-up shot. At first only a moving shadow crosses a quiet wall. After a brief pause, the character suddenly leans into frame from the side, notices the viewer, and breaks into a cheerful smile. Hair follows the quick movement and settles as the character holds the final expression. Keep the camera fixed. No captions.

overall_soundscape:
Soft footsteps, a brief rustle of clothing and a natural quiet laugh. No spoken dialogue.

non_diegetic_music:
A light playful musical accent on the reveal, then a soft finish.
"""


def clean_full_prompt(value):
    if not isinstance(value, str):
        raise ValueError("Full promptには文章を貼り付けてください。 / Full prompt must be text.")
    value = value.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    # A copied single Markdown fence is presentation, not part of the prompt.
    fence = re.fullmatch(r"```[^\n]*\n(.*?)\n```", value, re.S)
    if fence:
        value = fence.group(1).strip()
    if not value or not re.search(r"\w", value):
        raise ValueError("Full promptが空です。演出・動作・音をこの1欄へ入力してください。")
    if "\x00" in value:
        raise ValueError("Full prompt contains a NUL character; paste plain text.")
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(f"Full promptは{MAX_PROMPT_CHARS}文字以内にしてください（自動切り捨てはしません）。")
    return value


def validate_references(prompt, count):
    for kind, number in re.findall(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", prompt, re.I):
        if kind.lower() != "picture" or not 1 <= int(number) <= count:
            raise ValueError(
                f"<{kind} {number}> の参照がありません。画像は <Picture 1>〜<Picture {count}>。"
                "このワークフローは同一キャラの静止画専用です。"
            )


class PromptBudgetClip:
    """Per-call facade: validate real text tokens before expensive Qwen encoding."""
    def __init__(self, clip):
        self.clip = clip

    def tokenize(self, prompt, **kwargs):
        tokens = self.clip.tokenize(prompt, **kwargs)
        text_tokens = sum(isinstance(entry[0], (int, float)) for batches in tokens.values()
                          for batch in batches for entry in batch)
        if text_tokens > 4096:
            raise ValueError("Full promptが長すぎます（4096 text tokens上限）。内容を短くしてください。自動切り捨てはしません。")
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        return self.clip.encode_from_tokens_scheduled(tokens)


class MiniMaxH3FullPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.5}),
            "full_prompt": ("STRING", {"default": DEFAULT_FULL_PROMPT, "multiline": True,
                                        "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "length")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = "全文を1欄へ。4欄への分割・モード切り替え不要。音・BGMも全文に含めます。No LLM/API."

    def build(self, duration_seconds, full_prompt):
        seconds = float(duration_seconds)
        if not math.isfinite(seconds) or not 5 <= seconds <= 15:
            raise ValueError("Choose 5–15 seconds; start with 5 for a VRAM check.")
        frames = round(seconds * 24)
        return clean_full_prompt(full_prompt), frames + (5 - frames) % 17


def image_paths(value, root):
    try:
        names = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("参照画像リストが壊れています。画像を追加し直してください。") from exc
    if not isinstance(names, list) or not 1 <= len(names) <= MAX_IMAGES:
        raise ValueError(f"同じキャラの画像を1〜{MAX_IMAGES}枚追加してください。")
    root = Path(root).resolve()
    paths = []
    for name in names:
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            raise ValueError("Invalid reference image path.")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("参照画像が見つかりません。Podを作り直した場合は再アップロードしてください。")
        if path in paths:
            raise ValueError("同じ画像が重複しています。複製では情報は増えないので1枚にしてください。")
        paths.append(path)
    return paths


def reference_sizes(sizes, max_dimension, max_tokens):
    if not 256 <= max_dimension <= 1536 or not 256 <= max_tokens <= 4096:
        raise ValueError("Reference limits: long edge 256–1536px; total tokens 256–4096.")
    if not 1 <= len(sizes) <= MAX_IMAGES or any(min(w, h) < 32 for w, h in sizes):
        raise ValueError("Use 1–8 images, each at least 32px on both axes.")

    def fit(scale):
        return [(max(32, int(w * min(1., max_dimension / max(w, h)) * scale) // 32 * 32),
                 max(32, int(h * min(1., max_dimension / max(w, h)) * scale) // 32 * 32))
                for w, h in sizes]

    planned = fit(1.)
    cost = sum((w // 32) * (h // 32) for w, h in planned)
    if cost > max_tokens:
        scale = math.sqrt(max_tokens / cost)
        planned = fit(scale)
        while sum(w * h // 1024 for w, h in planned) > max_tokens:
            scale *= .95
            planned = fit(scale)
    return planned


class MiniMaxH3RefModImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images_json": ("STRING", {"default": "[]", "multiline": True}),
            "max_dimension": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 32}),
            "max_total_tokens": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256}),
        }}

    RETURN_TYPES = ("H3_CHARACTER_IMAGES",)
    RETURN_NAMES = ("references",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = "同一キャラを1〜8枚。追加/削除/順序変更可能。別々の縦横比を維持、合計参照量を制限。"

    @classmethod
    def IS_CHANGED(cls, images_json, **kwargs):
        import folder_paths
        digest = hashlib.sha256()
        try:
            for path in image_paths(images_json, folder_paths.get_input_directory()):
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        except ValueError:
            return float("nan")
        return digest.hexdigest()

    def load(self, images_json, max_dimension=1024, max_total_tokens=2048):
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        paths = image_paths(images_json, folder_paths.get_input_directory())
        # Read dimensions before allocating full image tensors. Do not treat GIFs as videos.
        sizes = []
        for path in paths:
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("参照画像は40MP以下へ縮小してからアップロードしてください。")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("動画/アニメーションではなく静止画を使ってください。")
                sizes.append(ImageOps.exif_transpose(source).size)
        plans = reference_sizes(sizes, max_dimension, max_total_tokens)
        images = []
        for path, size in zip(paths, plans):
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGBA")
                white = Image.new("RGBA", rgb.size, "white")
                rgb = Image.alpha_composite(white, rgb).convert("RGB").resize(size, Image.Resampling.LANCZOS)
                images.append(torch.from_numpy(np.array(rgb).astype(np.float32) / 255.).unsqueeze(0))
        return ({"images": images, "names": [p.name for p in paths],
                 "budget": max_total_tokens, "original_sizes": sizes},)


def refmod_modules():
    """Load only upstream core + serializer, not unrelated UI routes/global hooks."""
    root = Path(os.environ.get("MINIMAX_H3_REFMOD_ROOT", Path(__file__).with_name("refmod_vendor")))
    if not (root / "REVISION").is_file() or (root / "REVISION").read_text().strip() != REFMOD_COMMIT:
        raise RuntimeError("Pinned RefMod runtime is missing. Use the updated RunPod Docker image.")
    name = "_minimax_h3_refmod_vendor"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(name + ".core"), importlib.import_module(name + ".bundle")


class MiniMaxH3CreateCharacterRefMod:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"references": ("H3_CHARACTER_IMAGES",), "vae": ("VAE",)}}

    RETURN_TYPES = ("H3_CHARACTER_REFMOD", "H3_REF_MODS")
    RETURN_NAMES = ("character", "mods")
    FUNCTION = "create"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = "Full Reference: 各画像を別々にVAE encode。顔用の強い圧縮なし。入力が同じならComfyUIが再利用。"

    def create(self, references, vae):
        import torch
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Connect the MiniMax H3 video VAE, not an audio or image-model VAE.")
        core, _ = refmod_modules()
        mods = []
        with torch.inference_mode():
            for index, image in enumerate(references["images"]):
                z = vae.encode(image).detach().cpu().clone()
                if z.ndim != 5 or tuple(z.shape[:3]) != (1, 24, 1) or any(n % 2 for n in z.shape[-2:]):
                    raise ValueError(f"Unexpected H3 still-image latent layout: {tuple(z.shape)}")
                mod = core.H3RefMod(name=f"character_picture_{index+1}", kind="image", latent=z,
                                   latent_h=z.shape[-2], latent_w=z.shape[-1], mode="encode",
                                   source="image", source_shape=f"1x{z.shape[-2]}x{z.shape[-1]}",
                                   concept_type="identity")
                mods.append((mod, 1.0))
        if sum(mod.token_count for mod, _ in mods) > references["budget"]:
            raise ValueError("Encoded RefMod exceeds the reference token budget; reduce reference size.")
        return ({**references, "mods": mods}, mods)


class MiniMaxH3CharacterRefModR2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "character": ("H3_CHARACTER_REFMOD",),
            "prompt": ("STRING", {"forceInput": True}),
            "width": ("INT", {"default": 480, "min": 32, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 864, "min": 32, "max": 2048, "step": 32}),
            "length": ("INT", {"default": 124, "min": 5, "max": 362, "step": 17}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "LATENT", "reference_map")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = "Native H3 text/vision conditioning + RefMod exactly once. No first frame, no global patches."

    def encode(self, clip, character, prompt, width, height, length):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        prompt = clean_full_prompt(prompt)
        images, mods = character["images"], character["mods"]
        validate_references(prompt, len(images))
        if len(images) != len(mods) or not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError("RefMod images/payload count mismatch.")
        if min(width, height) < 32 or width % 32 or height % 32 or not 5 <= length <= 362 or length % 17 != 5:
            raise ValueError("Use 32px image dimensions and the H3 17n+5 frame grid (up to 15s).")
        refs = {f"ref_image_{i}": img for i, img in enumerate(images)}
        # The original bounded pixels go to Qwen. Omitting vae prevents a second
        # encode/duplicate payload. The saved full-resolution RefMods go to DiT.
        result = MiniMaxH3ReferenceToVideo.execute(PromptBudgetClip(clip), prompt, width, height, length,
                                                  ref_image_size="max", vae=None, ref_images=refs).result
        blocks = [mod.ref_block(1.0) for mod, _ in mods]
        positive = []
        for embedding, metadata in result[0]:
            if "minimax_token_tags" not in metadata:
                raise ValueError("Use the H3 Qwen text encoder; minimax_token_tags are missing.")
            if metadata.get("minimax_refs"):
                raise ValueError("RefMod would be applied twice; check the conditioning path.")
            positive.append([embedding, {**metadata, "minimax_refs": blocks}])
        mapping = "\n".join(f"<Picture {i+1}> = {name} (same character)" for i, name in enumerate(character["names"]))
        return positive, result[1], mapping


class MiniMaxH3SaveCharacterRefMod:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mods": ("H3_REF_MODS",),
                              "filename_prefix": ("STRING", {"default": "character"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Character"
    DESCRIPTION = "互換RefModをoutput/refmodsへ保存。永続なしPodを削除する前にファイルをダウンロード。"

    def save(self, mods, filename_prefix="character"):
        import folder_paths
        import torch
        _, bundle = refmod_modules()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", filename_prefix):
            raise ValueError("RefMod filename: use 1–64 letters, numbers, _ or -.")
        digest = hashlib.sha256()
        for mod, _ in mods:
            digest.update(json.dumps(mod.metadata(), sort_keys=True).encode())
            digest.update(mod.latent.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        name = f"{filename_prefix}_{digest.hexdigest()[:16]}"
        path = Path(folder_paths.get_output_directory()) / "refmods" / name
        saved = bundle.save_bundle(str(path), name, mods)
        return {"ui": {"refmod_files": [{"filename": path.name + ".safetensors", "subfolder": "refmods", "type": "output"}]},
                "result": (saved,)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MiniMaxH3FullPrompt, MiniMaxH3RefModImages, MiniMaxH3CreateCharacterRefMod,
    MiniMaxH3CharacterRefModR2V, MiniMaxH3SaveCharacterRefMod,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FullPrompt": "Full prompt / 全文はここ1欄",
    "MiniMaxH3RefModImages": "Character images / 同一キャラ1〜8枚",
    "MiniMaxH3CreateCharacterRefMod": "Create Character RefMod / Full Reference",
    "MiniMaxH3CharacterRefModR2V": "Character RefMod → R2VA",
    "MiniMaxH3SaveCharacterRefMod": "Save Character RefMod / ダウンロード",
}
