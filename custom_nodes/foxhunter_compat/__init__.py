"""Only the two Easy-Use node IDs consumed by FoxFur Seed Hunter.

This avoids installing the entire Easy-Use pack (and its unrelated diffusion
dependencies) into the isolated FoxFur container.
"""

class FoxEasySeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 2**64 - 1})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "doit"
    CATEGORY = "FoxHunter/compat"

    def doit(self, seed=0):
        return (seed,)


class FoxEasyIfElse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"boolean": ("BOOLEAN", {"default": True})},
                "optional": {"on_true": ("*", {"lazy": True}),
                             "on_false": ("*", {"lazy": True})}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("*",)
    FUNCTION = "choose"
    CATEGORY = "FoxHunter/compat"

    @classmethod
    def check_lazy_status(cls, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    @classmethod
    def choose(self, boolean=True, on_true=None, on_false=None):
        return (on_true if boolean else on_false,)


NODE_CLASS_MAPPINGS = {"easy seed": FoxEasySeed, "easy ifElse": FoxEasyIfElse}
NODE_DISPLAY_NAME_MAPPINGS = {
    "easy seed": "EasySeed (FoxHunter compatibility)",
    "easy ifElse": "If else (FoxHunter compatibility)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
