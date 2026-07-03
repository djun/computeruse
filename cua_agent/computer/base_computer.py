"""Shared ComputerAdapter implementation backed by a vision pipeline + action engine.

OS adapters subclass :class:`DriverBackedComputerAdapter`, set ``self.vision`` and
``self.action_engine`` (plus ``platform_name`` / ``system_info`` / ``display``) in
their own ``__init__``, and implement ``run_health_checks``. The capture/hash/
grounding/execute delegations are identical across platforms and live here.
"""

from __future__ import annotations

from typing import Any

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.computer.drivers import BaseVisionPipeline
from cua_agent.utils.config import Settings


class DriverBackedComputerAdapter(ComputerAdapter):
    vision: BaseVisionPipeline
    action_engine: Any

    def run_health_checks(self, settings: Settings, logger: Any | None = None) -> None:
        raise NotImplementedError

    def capture_base64(self) -> str:
        return self.vision.capture_base64()

    def capture_with_hash(self) -> tuple[str, str]:
        return self.vision.capture_with_hash()

    def hash_base64(self, image_b64: str) -> str:
        return self.vision.hash_base64(image_b64)

    def hash_distance(self, hash_a: str | None, hash_b: str | None) -> int:
        return self.vision.hash_distance(hash_a, hash_b)

    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool:
        return self.vision.has_changed(previous_b64, current_b64, threshold=threshold)

    def structural_similarity(self, previous_b64: str, current_b64: str) -> float | None:
        return self.vision.structural_similarity(previous_b64, current_b64)

    def detect_ui_elements(self, image_b64: str) -> list[dict]:
        return self.vision.detect_ui_elements(image_b64)

    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult:
        return self.action_engine.accessibility_driver.get_active_window_tree(max_depth=max_depth)

    def execute(self, action: dict) -> ActionResult:
        return self.action_engine.execute(action)
