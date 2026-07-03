from __future__ import annotations

import subprocess
import time

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.action_engine_base import SharedActionEngine
from cua_agent.grounding.grounding_model import GroundingModelClient
from cua_agent.computer.drivers import (
    BaseAccessibilityDriver,
    BaseBrowserDriver,
    BaseHIDDriver,
    BaseSemanticDriver,
    BaseShellDriver,
    BaseVisionPipeline,
)
from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from windows_cua_agent.drivers.accessibility_driver import AccessibilityDriver
from windows_cua_agent.drivers.browser_driver import BrowserDriver
from windows_cua_agent.drivers.hid_driver import HIDDriver
from windows_cua_agent.drivers.semantic_driver import SemanticDriver
from windows_cua_agent.drivers.shell_driver import ShellDriver
from windows_cua_agent.utils.windows_integration import (
    get_display_info,
    get_foreground_process_image_name,
    get_foreground_window_title,
)


class ActionEngine(SharedActionEngine):
    """Windows action routing: UIA + SendInput HID, with Cyborg (CDP-less) browser fallback."""

    # Windows leaves the policy HITL reason empty when the policy gives none, and
    # drives Chrome by default.
    POLICY_HITL_DEFAULT = ""
    DEFAULT_BROWSER_APP = "Chrome"

    def __init__(
        self,
        settings: Settings,
        policy_engine: PolicyEngine,
        vision_pipeline: BaseVisionPipeline | None = None,
    ) -> None:
        self.settings = settings
        self.policy_engine = policy_engine
        self.display = get_display_info()
        self.hid_driver: BaseHIDDriver = HIDDriver(settings)
        self.semantic_driver: BaseSemanticDriver = SemanticDriver(settings)
        shell_allowlist = getattr(policy_engine, "shell_allowlist", None)
        self.shell_driver: BaseShellDriver = ShellDriver(settings, allowed_commands=shell_allowlist)
        self.accessibility_driver: BaseAccessibilityDriver = AccessibilityDriver(settings)
        self.browser_driver: BaseBrowserDriver = BrowserDriver(settings)
        self.vision_pipeline = vision_pipeline
        self.grounding_model = GroundingModelClient(settings)
        self.logger = get_logger(__name__, level=settings.log_level)

    def _primary_modifier_key(self) -> str:
        return "ctrl"

    def _clipboard_read(self) -> str:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"],
            text=True,
        )

    def _clipboard_write(self, content: str) -> None:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
            input=content,
            text=True,
            check=True,
        )

    def _clipboard_clear(self) -> None:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Set-Clipboard -Value ''"],
            check=True,
        )

    def _platform_enrich(self, enriched: dict) -> None:
        """Attach the active process name to `bundle_id` for Windows policy rules."""
        try:
            exe = get_foreground_process_image_name()
            if exe:
                enriched.setdefault("bundle_id", exe)
                enriched["active_exe"] = exe
        except Exception:
            pass

        try:
            title = get_foreground_window_title()
            if title:
                enriched["active_window_title"] = title
        except Exception:
            pass

    def _extra_hitl_reason(self, action: dict) -> str:
        return "windows high-risk heuristic" if self._requires_hitl(action) else ""

    def _requires_hitl(self, action: dict) -> bool:
        # UAC / elevation prompts.
        exe = (action.get("bundle_id") or "").lower()
        title = (action.get("active_window_title") or "").lower()
        if exe == "consent.exe" or "user account control" in title:
            return True

        # Dangerous shell operations (best-effort, heuristic).
        if action.get("type") == "sandbox_shell":
            cmd = (action.get("cmd") or action.get("command") or "").lower()
            destructive = ["rd /s", "rmdir /s", "del /s", "remove-item", "rm -rf", "format "]
            if any(pat in cmd for pat in destructive):
                return True
            script_exts = [".ps1", ".bat", ".vbs"]
            if any(ext in cmd for ext in script_exts):
                return True
        return False

    def _execute_browser(self, action: dict) -> ActionResult:
        browser_result = self.browser_driver.execute_browser_action(action)
        if browser_result.success:
            return browser_result

        if self.settings.windows_cyborg_mode and self._looks_like_cdp_unavailable(browser_result.reason):
            fallback = self._cyborg_fallback_for_browser_action(action)
            if fallback is not None:
                return fallback
            # Non-actionable browser ops (DOM/JS) should be retried via computer tools.
            return ActionResult(
                success=True,
                reason=(
                    f"CDP unavailable; skipped browser.{action.get('command')} "
                    "(use computer/inspect_ui + HID/Phantom Mode)"
                ),
                metadata={"cdp_unavailable": True, **(browser_result.metadata or {})},
            )

        return browser_result

    def _looks_like_cdp_unavailable(self, reason: str) -> bool:
        """
        Best-effort classifier for Chrome DevTools Protocol unavailability.

        Chrome 136+ can refuse to expose the CDP listener for the default profile, which
        surfaces as connection errors, empty /json listings, websocket upgrade failures,
        or timeouts. In these cases we should degrade to "Cyborg" (UIA/HID/Vision) mode.
        """
        r = (reason or "").lower()
        if not r:
            return True

        needles = [
            "connection refused",
            "connectex",
            "actively refused",
            "timed out",
            "timeout waiting for",
            "websocket upgrade failed",
            "no response",
            "socket closed",
            "no page target found",
            "cdp websocket not connected",
            "urlopen error",
            "failed to establish a new connection",
        ]
        return any(n in r for n in needles)

    def _cyborg_fallback_for_browser_action(self, action: dict) -> ActionResult | None:
        """
        Convert certain browser ops into equivalent UI-level interactions.

        This keeps Windows automation functional when CDP is blocked (e.g., Chrome 136+
        default profile restrictions) by using the same interfaces a human uses.
        """
        cmd = (action.get("command") or "").strip()

        if cmd == "navigate":
            url = (action.get("url") or "").strip()
            if not url:
                return ActionResult(success=False, reason="navigate requires url")
            macro = [
                {"type": "key", "keys": ["ctrl", "l"]},
                {"type": "wait", "seconds": 0.15},
                {"type": "type", "text": url},
                {"type": "key", "keys": ["enter"]},
            ]
            res = self._run_macro_actions(macro)
            if res.success:
                return ActionResult(success=True, reason="CDP unavailable; navigated via Cyborg macro", metadata=res.metadata)
            return ActionResult(
                success=False,
                reason=f"CDP unavailable; Cyborg navigate macro failed: {res.reason}",
                metadata=res.metadata,
            )

        if cmd in {"go_back", "go_forward", "reload"}:
            if cmd == "go_back":
                keys = ["alt", "left"]
            elif cmd == "go_forward":
                keys = ["alt", "right"]
            else:
                keys = ["ctrl", "r"]
            res = self.execute({"type": "key", "keys": keys})
            if res.success:
                return ActionResult(success=True, reason=f"CDP unavailable; {cmd} via Cyborg hotkey")
            return ActionResult(success=False, reason=f"CDP unavailable; {cmd} hotkey failed: {res.reason}")

        return None

    def _open_app(self, action: dict) -> ActionResult:
        app_name = action.get("app_name", "")
        self.logger.info("Executing open_app for: %s", app_name)

        if self.settings.enable_semantic:
            res = self.semantic_driver.execute({"command": "open_app", "app_name": app_name})
            if res.success:
                return res
            res = self.semantic_driver.execute({"command": "focus_app", "app_name": app_name})
            if res.success:
                return res

        # Fall back to Start menu sequence
        res = self.hid_driver.press_keys(["win"])
        if not res.success:
            return res
        time.sleep(0.4)
        res = self.hid_driver.type_text(app_name)
        if not res.success:
            return res
        time.sleep(0.2)
        return self.hid_driver.press_keys(["enter"])
