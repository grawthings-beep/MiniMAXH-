"""Memory barriers for MiniMax H3 sampling-to-decode transitions."""

from __future__ import annotations

import gc


class MiniMaxH3VAEDecodeTiled:
    """Decode H3 video with its native tiler after unwrapping the AV latent pair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/Memory"
    DESCRIPTION = (
        "Decodes only the video member of MiniMax H3's nested video/audio latent "
        "using the VAE's native 256px spatial tiles and 17-frame temporal chunks."
    )

    def decode(self, vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        images = vae.decode_tiled(latent)
        if len(images.shape) == 5:
            images = images.reshape(
                -1,
                images.shape[-3],
                images.shape[-2],
                images.shape[-1],
            )
        return (images,)


class MiniMaxH3ReleaseVRAMLatent:
    """Unload diffusion/text models before the video and audio VAE decoders run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "release"
    CATEGORY = "MiniMax H3/Memory"
    DESCRIPTION = (
        "Passes the sampled latent through after unloading GPU models and clearing "
        "the CUDA allocator. Place it directly before VAE Decode (Tiled)."
    )

    def release(self, samples):
        from comfy import model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        gc.collect()
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3VAEDecodeTiled": MiniMaxH3VAEDecodeTiled,
    "MiniMaxH3ReleaseVRAMLatent": MiniMaxH3ReleaseVRAMLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3VAEDecodeTiled": "MiniMax H3 VAE Decode Tiled (Nested Safe)",
    "MiniMaxH3ReleaseVRAMLatent": "Release H3 VRAM Before Tiled VAE Decode",
}


__all__ = [
    "MiniMaxH3VAEDecodeTiled",
    "MiniMaxH3ReleaseVRAMLatent",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
