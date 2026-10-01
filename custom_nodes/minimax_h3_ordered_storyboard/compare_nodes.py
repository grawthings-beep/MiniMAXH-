"""One sampled AV latent, three serial finishing passes, disk-backed previews.

No full-resolution IMAGE output is cached by ComfyUI. Each branch is saved and
released before the next starts. A full decoded clip (and mosaic working copy)
still occupies CPU RAM: this is not constant-memory streaming.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import time
import uuid
from fractions import Fraction
from pathlib import Path

X2_VAE = "MiniMax-H3-X2-Detail-v1.safetensors"
STOCK_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
SPAN = "2xNomosUni_span_multijpg.safetensors"
ANIME = "2x-AnimeSharpV4_RCAN.safetensors"
VARIANTS = (("A_X2_VAE", X2_VAE), ("B_INT8_SPAN", SPAN), ("C_INT8_AnimeSharp", ANIME))


def _first(result):
    if hasattr(result, "result"):
        return result.result[0]
    if hasattr(result, "args"):
        return result.args[0]
    return result[0]


def _release():
    from comfy import model_management
    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache(True)


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode(samples, vae, variant):
    if variant == "A_X2_VAE":
        import nodes
        decoder = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3VAEDecodeFast")
        if decoder is None:
            raise RuntimeError("X2 decoder missing; use a comparison-enabled image.")
        return _first(decoder().decode(samples, vae, tiling=True, tile_size=256,
                                      tile_overlap=64, output_device="cpu", temporal_tiling=False))
    from .memory_nodes import MiniMaxH3VAEDecodeTiled
    return MiniMaxH3VAEDecodeTiled().decode(vae, samples)[0]


def _upscale_frames(images, name):
    """One frame on GPU at a time; keep only the final 2x clip on CPU."""
    import torch
    from comfy import model_management
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
    model = _first(UpscaleModelLoader.execute(name))
    try:
        if model.scale != 2:
            raise ValueError("Comparison requires an exact 2x upscale model")
        result = torch.empty((len(images), images.shape[1] * 2, images.shape[2] * 2, 3),
                             dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            for index in range(len(images)):
                model_management.throw_exception_if_processing_interrupted()
                frame = _first(ImageUpscaleWithModel.execute(model, images[index:index + 1]))
                if tuple(frame.shape[1:]) != tuple(result.shape[1:]):
                    raise ValueError("Upscale model returned the wrong dimensions")
                result[index].copy_(frame[0].detach().to(device="cpu", dtype=torch.float32))
                del frame
    finally:
        _release()
        del model
    return result


def _mosaic(images, enabled):
    from .mosaic_nodes import WanAutoMosaicVideo, MODEL_FILENAME, DEFAULT_CLASSES
    return WanAutoMosaicVideo().apply(images, MODEL_FILENAME, enabled, "JUST", 0.30,
                                     0.50, 0, 3, DEFAULT_CLASSES)[0]


def _save_video(images, audio, fps, path, metadata):
    from comfy_api.latest import InputImpl, Types
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)),
        bit_depth=8, color_space="sRGB")
    temporary = path.with_name(path.stem + ".partial.mp4")
    try:
        video.save_to(str(temporary), format=Types.VideoContainer("mp4"),
                      codec=Types.VideoCodec("h264"), crf=18, metadata=metadata)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class MiniMaxH3CompareUpscale:
    CATEGORY = "MiniMax H3/Comparison"
    FUNCTION = "compare"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_report",)
    DESCRIPTION = (
        "Sample once; serial X2 VAE / INT8+SPAN / INT8+AnimeSharp exports. Same audio, "
        "size and H.264 CRF18. Reference encoding is fixed to INT8. CPU RAM is still "
        "required for a decoded clip. AnimeSharp: Kim2091, CC-BY-NC-SA-4.0."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "standard_vae": ("VAE",), "audio": ("AUDIO",),
            "width": ("INT", {"default": 480, "min": 32}),
            "height": ("INT", {"default": 864, "min": 32}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "filename_prefix": ("STRING", {"default": "video/H3_Upscale_Compare"}),
            "mosaic_enabled": ("BOOLEAN", {"default": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        import folder_paths
        for group, name in (("vae", X2_VAE), ("upscale_models", SPAN), ("upscale_models", ANIME)):
            if not folder_paths.get_full_path(group, name):
                return f"Comparison model missing: {name}. Set H3_R2V_COMPARE=1 and restart."
        return True

    def compare(self, samples, standard_vae, audio, width, height, fps,
                filename_prefix, mosaic_enabled, prompt=None, extra_pnginfo=None):
        import torch
        import nodes
        import folder_paths
        from comfy import model_management
        expected = (height * 2, width * 2, 3)
        if width < 32 or height < 32 or not math.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError("Invalid comparison dimensions or frame rate")
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width * 2, height * 2)
        stem = f"{name}_{counter:05}_{uuid.uuid4().hex[:8]}"
        report_path = Path(folder) / f"{stem}_report.json"
        metadata = dict(extra_pnginfo or {})
        if prompt is not None:
            metadata["prompt"] = prompt
        # All outputs use exactly the same decoded audio tensor and fps.
        report = {"schema_version": 1, "status": "running", "fps": fps,
                  "reference_vae": STOCK_VAE, "same_sampled_latent": True,
                  "same_audio": True, "width": width * 2, "height": height * 2,
                  "mosaic_enabled": bool(mosaic_enabled), "codec": "h264", "crf": 18,
                  "note": "Finishing comparison only; not an end-to-end reference-VAE comparison. "
                          "Timings include cold loads; GPU allocated peak is not total VRAM or RAM.",
                  "licenses": {"SPAN": "Helaman / CC-BY-4.0", "AnimeSharp": "Kim2091 / CC-BY-NC-SA-4.0"},
                  "variants": []}
        previews, frame_count = [], None
        try:
            for label, model_name in VARIANTS:
                model_management.throw_exception_if_processing_interrupted()
                _release()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                _sync()
                start = time.perf_counter()
                frames, decoder_vae = None, None
                try:
                    decoder_vae = nodes.VAELoader().load_vae(X2_VAE)[0] if label == "A_X2_VAE" else standard_vae
                    frames = _decode(samples, decoder_vae, label).detach().to(device="cpu", dtype=torch.float32)
                    _sync()
                    decoded_at = time.perf_counter()
                    decoder_vae = None
                    _release()
                    if label != "A_X2_VAE":
                        if tuple(frames.shape[1:]) != (height, width, 3):
                            raise ValueError("Stock decode size differs from the generation size")
                        frames = _upscale_frames(frames, model_name)
                    if tuple(frames.shape[1:]) != expected:
                        raise ValueError(f"{label}: expected {expected}, got {tuple(frames.shape[1:])}; refusing a misleading comparison")
                    if frame_count is not None and frame_count != len(frames):
                        raise ValueError("Comparison branches returned different frame counts")
                    frame_count = len(frames)
                    _sync()
                    upscaled_at = time.perf_counter()
                    frames = _mosaic(frames, mosaic_enabled)
                    mosaicked_at = time.perf_counter()
                    filename = f"{stem}_{label}.mp4"
                    _save_video(frames, audio, fps, Path(folder) / filename, metadata)
                    _sync()
                    finished_at = time.perf_counter()
                    row = {"label": label, "model": model_name, "frames": frame_count,
                           "filename": filename, "subfolder": subfolder, "type": "output",
                           "seconds": {"load_decode": decoded_at - start,
                                       "upscale_release": upscaled_at - decoded_at,
                                       "mosaic": mosaicked_at - upscaled_at,
                                       "encode": finished_at - mosaicked_at,
                                       "total": finished_at - start},
                           "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
                    report["variants"].append(row)
                    previews.append(row)
                    logging.info("[h3-compare] %s: %dx%d / %d frames / %.2fs", label, width * 2, height * 2, frame_count, row["seconds"]["total"])
                finally:
                    frames, decoder_vae = None, None
                    _release()
            report["status"] = "complete"
        except BaseException as error:
            report["status"] = "failed"
            report["error_type"] = type(error).__name__
            raise
        finally:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ui": {"comparison_files": previews,
                       "comparison_report": [{"filename": report_path.name, "subfolder": subfolder, "type": "output"}]},
                "result": (str(report_path),)}


NODE_CLASS_MAPPINGS = {"MiniMaxH3CompareUpscale": MiniMaxH3CompareUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3CompareUpscale": "H3 同一latent比較 · X2 VAE / SPAN / AnimeSharp"}
