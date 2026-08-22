"""Variable-length ordered keyframes for MiniMax H3 Director."""

from .storyboard import MiniMaxH3OrderedStoryboard
from .exporter import MiniMaxH3StoryExport2x


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3OrderedStoryboard": MiniMaxH3OrderedStoryboard,
    "MiniMaxH3StoryExport2x": MiniMaxH3StoryExport2x,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3OrderedStoryboard": "MiniMax H3 Ordered Storyboard",
    "MiniMaxH3StoryExport2x": "MiniMax H3 Story Export 2x (Memory Bounded)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
