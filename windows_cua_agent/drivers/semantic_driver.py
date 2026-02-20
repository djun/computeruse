from __future__ import annotations

import ctypes
import subprocess
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.drivers import BaseSemanticDriver
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class SemanticDriver(BaseSemanticDriver):
    """Best-effort semantic execution for Windows (focus app, basic intents)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        command = action.get("command")
        if command == "focus_app":
            return self._focus_app(action.get("app_name") or action.get("app"))
        if command == "focus_window":
            return self._focus_window(action.get("window_title"))
        if command == "open_app":
            return self._open_app(action.get("app_name") or action.get("app"))
        if command == "insert_text_at_cursor":
            return ActionResult(success=False, reason="unsupported semantic command")
        if command == "save_document":
            return ActionResult(success=False, reason="unsupported semantic command")

        self.logger.info("Semantic driver received unsupported command: %s", action)
        return ActionResult(success=False, reason="unsupported semantic command")

    def _focus_app(self, app_name: Optional[str]) -> ActionResult:
        return self._focus_window(app_name)

    def _focus_window(self, window_title: Optional[str]) -> ActionResult:
        if not window_title:
            return ActionResult(success=False, reason="window_title required for focus_window")
        needle = window_title.lower().strip()
        if not needle:
            return ActionResult(success=False, reason="window_title required for focus_window")

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
                return ActionResult(success=False, reason=f"no window title matched {window_title!r}")

            hwnd = matches[0]
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            return ActionResult(success=True, reason=f"focused {window_title}")
        except Exception as exc:
            return ActionResult(success=False, reason=f"focus_window failed: {exc}")

    def _open_app(self, app_name: Optional[str]) -> ActionResult:
        if not app_name:
            return ActionResult(success=False, reason="app_name required for open_app")
        token = app_name.strip()
        if not token:
            return ActionResult(success=False, reason="app_name required for open_app")

        try:
            shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
            result = int(shell32.ShellExecuteW(None, "open", token, None, None, 1))
            if result > 32:
                return ActionResult(success=True, reason=f"opened {token}")
        except Exception:
            pass

        try:
            safe_token = token.replace("'", "''")
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"Start-Process -FilePath '{safe_token}'"],
                check=True,
                capture_output=True,
                text=True,
            )
            return ActionResult(success=True, reason=f"opened {token}")
        except Exception as exc:
            return ActionResult(success=False, reason=f"open_app failed: {exc}")
