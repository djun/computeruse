from __future__ import annotations

from cua_agent.computer.vision_pipeline_base import SharedVisionPipeline
from cua_agent.utils.config import Settings
from macos_cua_agent.utils.macos_integration import get_display_info


class VisionPipeline(SharedVisionPipeline):
    """macOS screen capture and visual grounding (CoreGraphics via mss)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, get_display_info=get_display_info, with_cursor=True)
