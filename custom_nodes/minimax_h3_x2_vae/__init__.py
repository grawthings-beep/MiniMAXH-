"""Expose only the pinned upstream H3 decoder; no latent upscaler or web routes."""
from .vae_decode import MiniMaxH3VAEDecodeFast

NODE_CLASS_MAPPINGS = {"MiniMaxH3VAEDecodeFast": MiniMaxH3VAEDecodeFast}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3VAEDecodeFast": "MiniMax H3 VAE Decode (fast)"}
