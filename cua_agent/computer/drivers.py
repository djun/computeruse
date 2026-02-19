from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from cua_agent.agent.state_manager import ActionResult


class BaseAccessibilityDriver(ABC):
    """Interface for platform accessibility/semantic UI drivers."""

    @abstractmethod
    def get_active_window_tree(self, max_depth: int = 5) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def probe_element(self, x: float, y: float, radius: float = 0.0) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def perform_action_at(self, x: float, y: float, action: str) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def set_text_element_value(self, x: float, y: float, value: str) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def set_focused_element_value(self, value: str) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def get_focused_app_name(self) -> str | None:
        raise NotImplementedError


class BaseHIDDriver(ABC):
    """Interface for low-level HID input drivers."""

    @abstractmethod
    def move(self, x: float, y: float) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def left_click(self, x: float, y: float) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def right_click(self, x: float, y: float) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def double_click(self, x: float, y: float) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def type_text(self, text: str) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def press_keys(self, keys: Iterable[str]) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def scroll(self, clicks: int, axis: str = "vertical") -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def drag_and_drop(
        self,
        x: float,
        y: float,
        tx: float,
        ty: float,
        duration: float = 0.5,
        hold_delay: float = 0.0,
    ) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def select_area(
        self,
        x: float,
        y: float,
        tx: float,
        ty: float,
        duration: float = 0.4,
        hold_delay: float = 0.0,
    ) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def hover(self, x: float, y: float, duration: float = 1.0) -> ActionResult:
        raise NotImplementedError


class BaseSemanticDriver(ABC):
    """Interface for high-level semantic execution drivers."""

    @abstractmethod
    def execute(self, action: dict) -> ActionResult:
        raise NotImplementedError


class BaseBrowserDriver(ABC):
    """Interface for browser automation drivers."""

    @abstractmethod
    def execute_browser_action(self, action: dict) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def get_current_url(self, app_name: str) -> str | None:
        raise NotImplementedError


class BaseShellDriver(ABC):
    """Interface for sandboxed shell execution drivers."""

    @abstractmethod
    def execute(self, action: dict) -> ActionResult:
        raise NotImplementedError


class BaseVisionPipeline(ABC):
    """Interface for screen capture and visual grounding pipelines."""

    @abstractmethod
    def capture_base64(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def capture_with_hash(self) -> tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def hash_base64(self, image_b64: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def hash_distance(self, hash_a: str | None, hash_b: str | None) -> int:
        raise NotImplementedError

    @abstractmethod
    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool:
        raise NotImplementedError

    @abstractmethod
    def structural_similarity(self, previous_b64: str, current_b64: str) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def detect_ui_elements(self, image_b64: str) -> list[dict]:
        raise NotImplementedError
