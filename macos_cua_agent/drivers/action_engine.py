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
from macos_cua_agent.drivers.accessibility_driver import AccessibilityDriver
from macos_cua_agent.drivers.browser_driver import BrowserDriver
from macos_cua_agent.drivers.hid_driver import HIDDriver
from macos_cua_agent.drivers.semantic_driver import SemanticDriver
from macos_cua_agent.drivers.shell_driver import ShellDriver
from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info


class ActionEngine(SharedActionEngine):
    """macOS action routing: Accessibility (AXUIElement) + pyautogui/AppleScript HID."""

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
        self.shell_driver: BaseShellDriver = ShellDriver(settings)
        self.accessibility_driver: BaseAccessibilityDriver = AccessibilityDriver(settings)
        self.browser_driver: BaseBrowserDriver = BrowserDriver(settings)
        self.vision_pipeline = vision_pipeline
        self.grounding_model = GroundingModelClient(settings)
        self.logger = get_logger(__name__, level=settings.log_level)

    def _primary_modifier_key(self) -> str:
        return "command"

    def _clipboard_read(self) -> str:
        return subprocess.check_output(["pbpaste"]).decode("utf-8")

    def _clipboard_write(self, content: str) -> None:
        subprocess.run(["pbcopy"], input=content.encode("utf-8"), check=True)

    def _clipboard_clear(self) -> None:
        subprocess.run(["pbcopy"], input=b"", check=True)

    def _open_app(self, action: dict) -> ActionResult:
        app_name = action.get("app_name", "")
        self.logger.info("Executing open_app for: %s", app_name)

        if self.settings.enable_semantic:
            res = self.semantic_driver.execute({"command": "open_app", "app_name": app_name})
            if res.success:
                return res
            res = self.semantic_driver.execute({"command": "focus_app", "app_name": app_name})
            if res.success:
                focused = self.accessibility_driver.get_focused_app_name()
                if focused and app_name.lower() in focused.lower():
                    return res
                self.logger.info(
                    "Semantic focus reported success but focused app was %s; falling back to Spotlight",
                    focused or "unknown",
                )

        self.logger.info("Semantic focus failed or disabled; falling back to Spotlight HID sequence")

        # 2. Open Spotlight
        res = self.hid_driver.press_keys(["command", "space"])
        if not res.success:
            return res
        time.sleep(0.5)  # Wait for Spotlight animation

        # 3. Type App Name
        res = self.hid_driver.type_text(app_name)
        if not res.success:
            return res
        time.sleep(0.3)  # Wait for search results

        # 4. Press Enter
        return self.hid_driver.press_keys(["enter"])
