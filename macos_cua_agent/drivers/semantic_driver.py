from __future__ import annotations

import subprocess
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.drivers import BaseSemanticDriver
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class SemanticDriver(BaseSemanticDriver):
    """Semantic execution via AppleScript/JXA for high-level app intents."""

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
            return self._insert_text(action.get("text", ""))
        if command == "save_document":
            return self._save_document()

        self.logger.info("Semantic driver received unsupported command: %s", action)
        return ActionResult(success=False, reason="unsupported semantic command")

    def _focus_app(self, app_name: Optional[str]) -> ActionResult:
        if not app_name:
            return ActionResult(success=False, reason="app_name required for focus_app")
        script = """
        on run argv
            set targetApp to item 1 of argv
            tell application targetApp to activate
        end run
        """
        return self._run_osascript(script, f"focus {app_name}", [app_name])

    def _focus_window(self, window_title: Optional[str]) -> ActionResult:
        if not window_title:
            return ActionResult(success=False, reason="window_title required for focus_window")
        token = window_title.strip()
        if not token:
            return ActionResult(success=False, reason="window_title required for focus_window")

        script = """
        on run argv
            set targetTitle to item 1 of argv
            tell application "System Events"
                set procList to application processes whose background only is false
                repeat with procRef in procList
                    try
                        set procName to name of procRef
                        set winList to windows of procRef
                        repeat with winRef in winList
                            try
                                set winName to name of winRef
                                if winName contains targetTitle then
                                    tell application procName to activate
                                    return procName
                                end if
                            end try
                        end repeat
                    end try
                end repeat
            end tell
            error "window not found"
        end run
        """
        return self._run_osascript(script, f"focus_window {token}", [token])

    def _open_app(self, app_name: Optional[str]) -> ActionResult:
        if not app_name:
            return ActionResult(success=False, reason="app_name required for open_app")
        token = app_name.strip()
        if not token:
            return ActionResult(success=False, reason="app_name required for open_app")
        try:
            completed = subprocess.run(
                ["open", "-a", token],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return ActionResult(success=True, reason=f"opened {token}")
            return ActionResult(
                success=False,
                reason=f"open_app failed: {completed.stderr.strip() or completed.stdout.strip()}",
            )
        except Exception as exc:
            return ActionResult(success=False, reason=f"open_app exception: {exc}")

    def _insert_text(self, text: str) -> ActionResult:
        if not text:
            return ActionResult(success=False, reason="no text provided")
        script = """
        on run argv
            set targetText to item 1 of argv
            tell application "System Events" to keystroke targetText
        end run
        """
        return self._run_osascript(script, "insert_text", [text])

    def _save_document(self) -> ActionResult:
        script = 'tell application "System Events" to keystroke "s" using {command down}'
        return self._run_osascript(script, "save_document")

    def _run_osascript(self, script: str, label: str, args: list[str] | None = None) -> ActionResult:
        try:
            cmd = ["osascript", "-e", script]
            if args:
                cmd.extend(args)
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return ActionResult(success=True, reason=f"{label} executed")
            return ActionResult(
                success=False,
                reason=f"{label} failed: {completed.stderr.strip() or completed.stdout.strip()}",
            )
        except Exception as exc:
            self.logger.warning("Semantic command %s failed: %s", label, exc)
            return ActionResult(success=False, reason=f"{label} exception: {exc}")
