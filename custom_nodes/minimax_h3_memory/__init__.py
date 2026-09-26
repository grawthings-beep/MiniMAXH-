"""Load only the pinned KJNodes H3 memory kernels (installed during image build)."""

from .minimax_nodes import MiniMaxChunkFeedForward, MiniMaxLowVRAMAttention

NODE_CLASS_MAPPINGS = {
    "MiniMaxChunkFeedForward": MiniMaxChunkFeedForward,
    "MiniMaxLowVRAMAttention": MiniMaxLowVRAMAttention,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxChunkFeedForward": "MiniMax H3 Chunk FeedForward",
    "MiniMaxLowVRAMAttention": "MiniMax H3 Low VRAM Attention",
}
