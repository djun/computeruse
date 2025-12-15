from __future__ import annotations

from typing import Any, Protocol

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.config import Settings


class ComputerAdapter(Protocol):
    """
    OS-specific implementation of the agent's "computer" capabilities.

    Adapter packages (e.g. `macos_cua_agent`, `windows_cua_agent`) are expected to
    implement this protocol and expose a `create_computer(settings: Settings)`.
    """

    platform_name: str
    system_info: str
    display: DisplayInfo

    def run_health_checks(self, settings: Settings, logger: Any | None = None) -> None: ...

    def capture_base64(self) -> str: ...
    def capture_with_hash(self) -> tuple[str, str]: ...
    def hash_base64(self, image_b64: str) -> str: ...
    def hash_distance(self, hash_a: str | None, hash_b: str | None) -> int: ...
    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool: ...
    def structural_similarity(self, previous_b64: str, current_b64: str) -> float | None: ...
    def detect_ui_elements(self, image_b64: str) -> list[dict]: ...

    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult: ...
    def execute(self, action: dict) -> ActionResult: ...

