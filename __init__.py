from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .lora_grid_node import MagikLoraGrid


class MagikLoraGridExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MagikLoraGrid]


async def comfy_entrypoint() -> MagikLoraGridExtension:
    return MagikLoraGridExtension()
