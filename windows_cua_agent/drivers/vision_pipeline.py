from __future__ import annotations

from cua_agent.computer.vision_pipeline_base import SharedVisionPipeline
from cua_agent.utils.config import Settings
from windows_cua_agent.utils.windows_integration import get_display_info


class VisionPipeline(SharedVisionPipeline):
    """Windows screen capture and visual grounding (mss without cursor)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, get_display_info=get_display_info, with_cursor=False)
