"""Only the two Easy-Use node IDs consumed by FoxFur Seed Hunter.

This avoids installing the entire Easy-Use pack (and its unrelated diffusion
dependencies) into the isolated FoxFur container.
"""

from comfy_api.latest import ComfyExtension, io


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


class FoxEasyIfElse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy ifElse",
            category="FoxHunter/compat",
            inputs=[
                io.Boolean.Input("boolean"),
                io.AnyType.Input("on_true", lazy=True),
                io.AnyType.Input("on_false", lazy=True),
            ],
            outputs=[io.AnyType.Output("*")],
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(kwargs["on_true"] if kwargs["boolean"] else kwargs["on_false"])


NODE_CLASS_MAPPINGS = {"easy seed": FoxEasySeed}
NODE_DISPLAY_NAME_MAPPINGS = {"easy seed": "EasySeed (FoxHunter compatibility)"}


class FoxHunterCompatExtension(ComfyExtension):
    async def get_node_list(self):
        return [FoxEasyIfElse]


async def comfy_entrypoint():
    return FoxHunterCompatExtension()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
