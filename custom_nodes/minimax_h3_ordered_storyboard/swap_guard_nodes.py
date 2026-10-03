"""CPU person masks, native H3 masked editing, and a source-copy rejection gate.

The gate detects near-reconstruction, NOT target identity. A reviewed mask and
manual review of the result are still required. No automatic paid retries.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

PERSON_MODEL = "yolo11s-seg.pt"
PERSON_SHA256 = "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3"
PERSON_BYTES = 20669228


def validate_source(source):
    import torch
    frames = source["frames"]
    if (frames.device.type != "cpu" or frames.dtype != torch.float32 or frames.ndim != 4
            or frames.shape[-1] != 3 or frames.shape[0] % 17 != 5
            or not 22 <= frames.shape[0] <= 362 or not torch.isfinite(frames).all()
            or float(frames.min()) < 0 or float(frames.max()) > 1):
        raise ValueError("Connect the bounded Source clip node (CPU RGB, valid H3 frame count).")
    if tuple(frames.shape[:3]) != (source["length"], source["height"], source["width"]):
        raise ValueError("Source dimensions do not match its frames.")
    return frames


def validate_mask(mask, frames):
    import torch
    if (mask.device.type != "cpu" or mask.ndim != 3
            or tuple(mask.shape) != tuple(frames.shape[:3])
            or not torch.isfinite(mask).all() or float(mask.min()) < 0 or float(mask.max()) > 1):
        raise ValueError("Mask must be CPU [frames, height, width], aligned to the prepared clip, in 0..1.")
    mask = mask.float().clone()
    if not (mask > .5).any():
        raise ValueError("No person region found. Do not run H3: review detection or provide an aligned mask.")
    return mask


def detect_people(frames, model_path, confidence=.25):
    """Every frame; single-person only. Do not silently pick another performer."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.is_file() or path.stat().st_size != PERSON_BYTES:
        raise RuntimeError("Verified swap person detector missing; use the updated image/startup manifest.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != PERSON_SHA256:
        raise RuntimeError("Swap person detector SHA256 mismatch; do not load this file.")
    # An absolute, already verified file prevents Ultralytics auto-download.
    detector = YOLO(str(path.resolve()), task="segment")
    masks, counts = [], []
    with torch.inference_mode():
        for frame in frames:
            rgb = (frame.numpy() * 255).round().clip(0, 255).astype(np.uint8)
            result = detector.predict(rgb[:, :, ::-1].copy(), classes=[0], device="cpu",
                                      conf=confidence, iou=.5, imgsz=640, retina_masks=True, verbose=False)[0]
            count = len(result.boxes)
            counts.append(count)
            if count > 1:
                raise ValueError("More than one person detected. Supply a target-only MASK or use original mode; no automatic target guessing.")
            if count and result.masks is not None:
                mask = F.interpolate(result.masks.data[:1, None].cpu().float(),
                                     size=frames.shape[1:3], mode="nearest")[0, 0]
                masks.append(mask)
            else:
                masks.append(torch.zeros(frames.shape[1:3]))
    del detector
    return torch.stack(masks), counts


def prepare_masks(mask, grow):
    """Grow frame masks and quantize to native 32px DiT cells + causal time bins.

Do not union all adjacent frames onto a correctly detected frame. Temporal
pooling here is exclusively the H3 latent-grid contract, not a tracking claim.
"""
    import torch
    import torch.nn.functional as F
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    from comfy_extras.nodes_minimax_h3 import video_latent_t

    n, h, w = mask.shape
    if h % 32 or w % 32 or not 0 <= grow <= 96:
        raise ValueError("32px canvas and grow 0..96 required.")
    grown = []
    for chunk in mask.split(8):
        x = chunk[:, None]
        if grow:
            x = F.max_pool2d(x, (1, 2*grow+1), stride=1, padding=(0, grow))
            x = F.max_pool2d(x, (2*grow+1, 1), stride=1, padding=(grow, 0))
        grown.append(x[:, 0])
    grown = torch.cat(grown)
    cells = F.max_pool2d(grown[:, None], 32, 32)[:, 0]
    counts = [FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(video_latent_t(n))]
    if sum(counts) != n:
        raise RuntimeError("H3 temporal mask contract changed; refusing unaligned editing.")
    chunks = torch.split(cells, counts)
    latent = torch.stack([chunk.amax(0) for chunk in chunks])
    latent = latent.repeat_interleave(2, -2).repeat_interleave(2, -1)[None, None]
    # Same regenerated cells used for generation and final compositing.
    pixels = latent[0, 0].repeat_interleave(torch.tensor(counts), dim=0)
    pixels = pixels.repeat_interleave(16, -2).repeat_interleave(16, -1)
    return grown, latent, pixels


def suppress_source_identity(frames, mask, retention, radius):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    if not math.isfinite(retention) or not 0 <= retention <= 1 or not 1 <= radius <= 128:
        raise ValueError("Reference retention must be 0..1; blur radius 1..128.")
    result = torch.empty_like(frames)
    for i, frame in enumerate(frames):
        if not mask[i].any() or retention == 1:
            result[i] = frame
            continue
        image = Image.fromarray((frame.numpy() * 255).round().clip(0, 255).astype(np.uint8))
        blur = torch.from_numpy(np.array(image.filter(ImageFilter.GaussianBlur(radius)), copy=True)).float()/255
        amount = mask[i, :, :, None] * (1-retention)
        result[i] = frame * (1-amount) + blur * amount
    return result


def build_edit_latent(reference_latent, empty_latent, edit):
    import torch
    import comfy.nested_tensor
    video, audio = empty_latent["samples"].unbind()
    mask = edit["latent_mask"]
    if tuple(reference_latent.shape) != tuple(video.shape) or tuple(mask.shape) != (1, 1, *video.shape[2:]):
        raise ValueError("Source latent / edit mask / output latent mismatch.")
    return {"samples": comfy.nested_tensor.NestedTensor((reference_latent, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mask, torch.ones_like(audio)))}


def copy_scores(original, generated, masks):
    """Per-frame, person-region correlation after resizing; invariant to exposure.

High correlation is evidence of unchanged pixels, not proof of unchanged
identity. Changed pixels likewise do not prove the requested identity.
"""
    import torch
    import torch.nn.functional as F
    if generated.shape[0] != original.shape[0] or generated.ndim != 4 or generated.shape[-1] != 3:
        raise ValueError("Generated frames do not align with the source clip.")
    scores = []
    for index, (source, output, mask) in enumerate(zip(original, generated, masks)):
        area = mask > .5
        if int(area.sum()) < 256:
            continue  # Empty/offscreen frames are not identity evidence.
        output = F.interpolate(output[None].movedim(-1, 1).float(), size=source.shape[:2],
                               mode="area")[0].movedim(0, -1)
        x, y = source[area].float(), output[area]
        x, y = x-x.mean(0), y-y.mean(0)
        denom = (x.square().sum()*y.square().sum()).sqrt()
        if denom < 1e-8:
            corr = 1.0 if torch.allclose(source[area], output[area], atol=.02) else 0.
        else:
            corr = float((x*y).sum()/denom)
        scores.append((index, corr))
    return scores


def reject_source_copy(scores, threshold=.985, fraction=.8):
    if not scores:
        raise ValueError("No visible target frames to check. Review the mask before generation.")
    unchanged = sum(score >= threshold for _, score in scores)
    ratio = unchanged / len(scores)
    if ratio >= fraction:
        raise ValueError(f"SWAP NOT CONFIRMED: {unchanged}/{len(scores)} target frames closely reproduce the source person. "
                         "MP4 export stopped. Review references/mask and lower source retention; do not call this a successful swap.")
    return ratio


class MiniMaxH3SwapTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("H3_SWAP_CLIP",),
            "mode": (["masked_replace", "original"],),
            "confidence": ("FLOAT", {"default": .25, "min": .05, "max": .9, "step": .05}),
            "grow": ("INT", {"default": 32, "min": 0, "max": 96, "step": 8}),
            "source_retention": ("FLOAT", {"default": .1, "min": 0., "max": 1., "step": .05}),
            "blur_radius": ("INT", {"default": 48, "min": 1, "max": 128}),
        }, "optional": {"target_mask": ("MASK",)}}
    RETURN_TYPES = ("H3_SWAP_EDIT", "STRING")
    RETURN_NAMES = ("edit", "mask_report")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Character Swap"
    OUTPUT_NODE = True

    def prepare(self, source, mode="masked_replace", confidence=.25, grow=32,
                source_retention=.1, blur_radius=48, target_mask=None):
        import torch
        import folder_paths
        from nodes import PreviewImage
        frames = validate_source(source)
        if mode == "original":
            return {"result": ({"mode": mode}, "Original successful pathway; no masked edit / copy gate." )}
        if mode != "masked_replace":
            raise ValueError("Unknown swap mode")
        if not math.isfinite(confidence) or not .05 <= confidence <= .9:
            raise ValueError("Confidence must be .05.. .9")
        if target_mask is None:
            raw, _counts = detect_people(frames, Path(folder_paths.models_dir)/"swap_detection"/PERSON_MODEL, confidence)
        else:
            raw = target_mask
        raw = validate_mask(raw, frames)
        grown, latent, pixels = prepare_masks(raw, grow)
        digest = hashlib.sha256()
        # Bind approval to the actual clip, mask and conditioning settings, not a
        # sticky checkbox that remains approved after uploading another clip.
        for frame in frames:
            digest.update(frame.numpy().tobytes())
        digest.update(raw.numpy().tobytes())
        digest.update(repr((grow, source_retention, blur_radius)).encode())
        approval = digest.hexdigest()[:12]
        visible = [i for i, mask in enumerate(raw) if bool((mask > .5).any())]
        absent = frames.shape[0]-len(visible)
        report = (f"Detected/masked: {len(visible)}/{len(frames)} frames; unmasked: {absent}. "
                  "Unmasked can mean offscreen OR missed detection. Inspect the ENTIRE preview before approving. "
                  "White/cyan marks the region allowed to change. Automatic masks do not guarantee identity. "
                  f"After review, paste this mask_approval code into the encoder: {approval}")
        # Full clip preview as frames: do not conceal detection gaps behind a few samples.
        overlay = frames.clone()
        cyan = torch.tensor([0., 1., 1.])
        overlay = overlay * (1-.45*pixels[..., None]) + cyan*(.45*pixels[..., None])
        preview = PreviewImage().save_images(overlay, filename_prefix="h3_swap_mask")
        edit = {"mode": mode, "frames": suppress_source_identity(frames, grown, source_retention, blur_radius),
                "latent_mask": latent, "pixel_mask": pixels, "raw_mask": raw,
                "shape": tuple(frames.shape), "report": report, "approval": approval}
        return {"ui": preview["ui"], "result": (edit, report)}


class MiniMaxH3SwapFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "source": ("H3_SWAP_CLIP",), "edit": ("H3_SWAP_EDIT",),
                             "reject_unchanged": ("BOOLEAN", {"default": True}),
                             "similarity_threshold": ("FLOAT", {"default": .985, "min": .9, "max": .999, "step": .001})}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "review_report")
    FUNCTION = "finish"
    CATEGORY = "MiniMax H3/Character Swap"

    def finish(self, images, source, edit, reject_unchanged=True, similarity_threshold=.985):
        import torch
        import torch.nn.functional as F
        frames = validate_source(source)
        if edit.get("mode") == "original":
            return images, "Original pathway; no automated identity verification."
        if tuple(images.shape[:1]) != tuple(frames.shape[:1]) or tuple(frames.shape) != edit["shape"]:
            raise ValueError("Generated output / source / edit do not align.")
        if not .9 <= similarity_threshold <= .999 or not math.isfinite(similarity_threshold):
            raise ValueError("Invalid source-copy threshold")
        images = images.detach().cpu().float()
        if not torch.isfinite(images).all():
            raise ValueError("Invalid generated pixels")
        scores = copy_scores(frames, images, edit["raw_mask"])
        if reject_unchanged:
            reject_source_copy(scores, similarity_threshold)
        result = torch.empty_like(images)
        for i, (frame, mask) in enumerate(zip(frames, edit["pixel_mask"])):
            h, w = images.shape[1:3]
            background = F.interpolate(frame[None].movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False)[0].movedim(0, -1)
            alpha = F.interpolate(mask[None, None], size=(h, w), mode="nearest")[0, 0, :, :, None]
            result[i] = images[i]*alpha + background*(1-alpha)
        return result, ("Source-copy check completed; NOT an identity pass. Review face, hair, clothes and re-entry. "
                        f"Checked {len(scores)} visible target frames; background outside edit cells restored from source.")


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (MiniMaxH3SwapTarget, MiniMaxH3SwapFinish)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SwapTarget": "Swap target · CPU mask / preview before H3",
    "MiniMaxH3SwapFinish": "Swap check · reject source copy / restore background",
}
