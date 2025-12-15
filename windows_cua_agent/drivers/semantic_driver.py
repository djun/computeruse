from __future__ import annotations

import ctypes
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class SemanticDriver:
    """Best-effort semantic execution for Windows (focus app, basic intents)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        command = action.get("command")
        if command == "focus_app":
            return self._focus_app(action.get("app_name") or action.get("app"))
        if command == "insert_text_at_cursor":
            return ActionResult(success=False, reason="unsupported semantic command")
        if command == "save_document":
            return ActionResult(success=False, reason="unsupported semantic command")

        self.logger.info("Semantic driver received unsupported command: %s", action)
        return ActionResult(success=False, reason="unsupported semantic command")

    def _focus_app(self, app_name: Optional[str]) -> ActionResult:
        if not app_name:
            return ActionResult(success=False, reason="app_name required for focus_app")
        needle = app_name.lower().strip()
        if not needle:
            return ActionResult(success=False, reason="app_name required for focus_app")

        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            matches: list[int] = []

            def _cb(hwnd, lparam):  # noqa: ANN001,ARG001
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = (buff.value or "").lower()
                if needle in title:
                    matches.append(int(hwnd))
                    return False
                return True

            user32.EnumWindows(EnumWindowsProc(_cb), 0)
            if not matches:
                return ActionResult(success=False, reason=f"no window title matched {app_name!r}")

            hwnd = matches[0]
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            return ActionResult(success=True, reason=f"focused {app_name}")
        except Exception as exc:
            return ActionResult(success=False, reason=f"focus_app failed: {exc}")

