from __future__ import annotations

import subprocess
from typing import Iterable, Optional, Tuple

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.coordinates import point_to_px
from cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info


class HIDDriver:
    """Executes low-level HID events. Defaults to dry-run unless ENABLE_HID is true."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_hid)
        self.display = get_display_info()
        self.pg = self._import_pyautogui() if self.enabled else None

    def _import_pyautogui(self) -> Optional[object]:
        try:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            return pyautogui
        except Exception as exc:
            self.logger.warning("pyautogui unavailable; HID execution disabled: %s", exc)
            self.enabled = False
            return None

    def move(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.moveTo(*self._to_px(x, y)), f"move({x},{y})")

    def left_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.click(*self._to_px(x, y)), f"left_click({x},{y})")

    def right_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.click(*self._to_px(x, y), button="right"), f"right_click({x},{y})")

    def double_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self.pg.doubleClick(*self._to_px(x, y)), f"double_click({x},{y})")

    def type_text(self, text: str) -> ActionResult:
        # Prefer AppleScript for typing to avoid stuck modifiers and flaky event injection on macOS.
        if self.enabled:
            try:
                script = """
                on run argv
                    set targetText to item 1 of argv
                    tell application "System Events" to keystroke targetText
                end run
                """
                subprocess.run(["osascript", "-e", script, text], check=True, capture_output=True)
                self.logger.info("HID action executed (via AppleScript): type:%r", text)
                return ActionResult(success=True)
            except Exception as exc:
                self.logger.warning("AppleScript typing failed, falling back to pyautogui: %s", exc)

        return self._perform(lambda: self.pg.write(text, interval=0.02), f"type:{text!r}")

    def press_keys(self, keys: Iterable[str]) -> ActionResult:
        combo = [self._normalize_key(k) for k in keys if k]
        return self._perform(lambda: self.pg.hotkey(*combo), f"hotkey:{'+'.join(combo)}")

    def scroll(self, clicks: int, axis: str = "vertical") -> ActionResult:
        if axis == "horizontal":
            # PyAutoGUI supports hscroll for horizontal scrolling
            return self._perform(lambda: self.pg.hscroll(clicks), f"hscroll:{clicks}")
        return self._perform(lambda: self.pg.scroll(clicks), f"scroll:{clicks}")

    def drag_and_drop(self, x: float, y: float, tx: float, ty: float, duration: float = 0.5, hold_delay: float = 0.0) -> ActionResult:
        def _action():
            sx, sy = self._to_px(x, y)
            dx, dy = self._to_px(tx, ty)
            self.pg.moveTo(sx, sy)
            if hold_delay > 0:
                import time
                time.sleep(hold_delay)
            # Use easeInOutQuad for "human-like" smoothing to avoid anti-bot detection
            self.pg.dragTo(dx, dy, duration=duration, button='left', mouseDownUp=True, tween=self.pg.easeInOutQuad)

        return self._perform(_action, f"drag({x},{y}->{tx},{ty})")
    
    def select_area(self, x: float, y: float, tx: float, ty: float, duration: float = 0.4, hold_delay: float = 0.0) -> ActionResult:
        def _action():
            sx, sy = self._to_px(x, y)
            ex, ey = self._to_px(tx, ty)
            self.pg.moveTo(sx, sy)
            if hold_delay > 0:
                import time
                time.sleep(hold_delay)
            # Explicit down/drag/up to simulate click-drag selection without moving objects
            self.pg.mouseDown(button="left")
            self.pg.dragTo(ex, ey, duration=duration, button="left")
            self.pg.mouseUp(button="left")

        return self._perform(_action, f"select_area({x},{y}->{tx},{ty})")

    def hover(self, x: float, y: float, duration: float = 1.0) -> ActionResult:
        def _action():
            self.pg.moveTo(*self._to_px(x, y))
            import time
            time.sleep(duration)

        return self._perform(_action, f"hover({x},{y}, {duration}s)")

    def _perform(self, func, label: str) -> ActionResult:
        if not self.enabled or not self.pg:
            self.logger.info("Dry-run HID: %s", label)
            return ActionResult(success=True, reason="dry-run")
        try:
            func()
            self.logger.info("HID action executed: %s", label)
            return ActionResult(success=True)
        except Exception as exc:
            self.logger.error("HID action failed: %s", exc)
            return ActionResult(success=False, reason=str(exc))

    def _to_px(self, x: float, y: float) -> Tuple[int, int]:
        return point_to_px(x, y, self.display.scale_factor)

    def _normalize_key(self, key: str) -> str:
        mapping = {
            "cmd": "command",
            "command": "command",
            "control": "ctrl",
            "option": "alt",
            "return": "enter",
        }
        lower = key.lower()
        return mapping.get(lower, lower)
