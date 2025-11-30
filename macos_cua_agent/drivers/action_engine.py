from __future__ import annotations

import math
import re
import subprocess
import time

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.drivers.accessibility_driver import AccessibilityDriver
from macos_cua_agent.drivers.browser_driver import BrowserDriver
from macos_cua_agent.drivers.hid_driver import HIDDriver
from macos_cua_agent.drivers.semantic_driver import SemanticDriver
from macos_cua_agent.drivers.shell_driver import ShellDriver
from macos_cua_agent.policies.policy_engine import PolicyEngine
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info


class ActionEngine:
    """Routes actions through policy evaluation, semantic path, or HID injection."""

    def __init__(self, settings: Settings, policy_engine: PolicyEngine) -> None:
        self.settings = settings
        self.policy_engine = policy_engine
        self.display = get_display_info()
        self.hid_driver = HIDDriver(settings)
        self.semantic_driver = SemanticDriver(settings)
        self.shell_driver = ShellDriver(settings)
        self.accessibility_driver = AccessibilityDriver(settings)
        self.browser_driver = BrowserDriver(settings)
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        decision = self.policy_engine.evaluate(action)
        if not decision.allowed:
            self.logger.warning("Action blocked by policy: %s", decision.reason)
            return ActionResult(success=False, reason=decision.reason)
        if decision.hitl_required:
            self.logger.warning("Action requires human confirmation: %s", action)
            return ActionResult(success=False, reason="human confirmation required")

        # Special-cased actions for loop control.
        if action.get("type") in ("noop", "capture_only"):
            return ActionResult(success=True, reason=action.get("reason", "noop"))
        if action.get("type") == "wait":
            seconds = float(action.get("seconds", 1))
            time.sleep(seconds)
            return ActionResult(success=True, reason=f"waited {seconds} seconds")

        self.logger.info("Executing action via %s: %s", action.get("execution", "hid"), action)
        execution_path = action.get("execution", "hid")
        action_type = action.get("type")

        if action_type == "inspect_ui":
            return self.accessibility_driver.get_active_window_tree()
            
        if action_type == "probe_ui":
            x = action.get("x")
            y = action.get("y")
            if x is None or y is None:
                return ActionResult(success=False, reason="probe_ui requires x,y coordinates")
            radius = float(action.get("radius") or 0.0)
            return self.accessibility_driver.probe_element(x, y, radius=radius)

        if action_type == "clipboard_op":
            return self._handle_clipboard(action)

        if action_type == "macro_actions":
            return self._run_macro_actions(action.get("actions") or [])
        
        if execution_path == "browser":
            return self.browser_driver.execute_browser_action(action)
        elif execution_path == "semantic" and self.settings.enable_semantic:
            result = self.semantic_driver.execute(action)
        elif execution_path == "shell":
            result = self.shell_driver.execute(action)
        else:
            result = self._execute_hid(action)

        self.logger.info("Action result: success=%s reason=%s", result.success, result.reason)
        return result

    def _handle_clipboard(self, action: dict) -> ActionResult:
        sub = action.get("sub_action")
        try:
            if sub == "read":
                content = subprocess.check_output(["pbpaste"]).decode("utf-8")
                sensitive, redacted = self._redact_clipboard_content(content)
                return ActionResult(
                    success=True,
                    reason="read clipboard (redacted)" if sensitive else "read clipboard",
                    metadata={"content": redacted, "redacted": sensitive},
                )
            elif sub == "write":
                content = action.get("content", "")
                subprocess.run(["pbcopy"], input=content.encode("utf-8"), check=True)
                return ActionResult(success=True, reason="wrote to clipboard")
            elif sub == "clear":
                subprocess.run(["pbcopy"], input=b"", check=True)
                return ActionResult(success=True, reason="cleared clipboard")
        except Exception as e:
            return ActionResult(success=False, reason=f"clipboard op failed: {e}")
        return ActionResult(success=False, reason=f"unknown clipboard sub_action: {sub}")

    def _redact_clipboard_content(self, content: str) -> tuple[bool, str]:
        """
        Lightweight secret detector to avoid leaking sensitive clipboard contents back to the model.
        """
        if not content:
            return False, content

        secret_patterns = [
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            r"AKIA[0-9A-Z]{16}",  # AWS Access Key
            r"(?i)eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9._-]+",  # JWT-like
            r"(?i)(api_key|secret|token|password)[=:]\s*[A-Za-z0-9\/+=_-]{8,}",
        ]
        for pat in secret_patterns:
            if re.search(pat, content):
                return True, "<REDACTED>"

        # Entropy heuristic for long opaque strings
        if len(content) >= 32 and self._shannon_entropy(content) > 4.0:
            return True, "<REDACTED>"

        return False, content

    def _shannon_entropy(self, data: str) -> float:
        freq = {ch: data.count(ch) for ch in set(data)}
        length = len(data) or 1
        return -sum((count / length) * math.log2(count / length) for count in freq.values())

    def _execute_hid(self, action: dict) -> ActionResult:
        action_type = action.get("type")
        x = action.get("x")
        y = action.get("y")
        phantom_mode = action.get("phantom_mode")
        # Default to phantom when we have semantic grounding (element_id) unless explicitly disabled
        if phantom_mode is None and action.get("element_id") is not None:
            phantom_mode = True
        phantom_mode = bool(phantom_mode)

        # Phantom Mode Check for Clicks
        if phantom_mode and action_type in ("left_click", "right_click", "double_click") and x is not None and y is not None:
            self.logger.info("Attempting Phantom Mode action for %s at (%s, %s)", action_type, x, y)
            if action_type == "left_click":
                res = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if res.success:
                    return res
                self.logger.info("Phantom left_click failed, falling back to physical HID")
            elif action_type == "right_click":
                # Try context-menu invocation if available; fall back to AXPress.
                res = self.accessibility_driver.perform_action_at(x, y, "AXShowMenu")
                if res.success:
                    return res
                res = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if res.success:
                    return ActionResult(success=True, reason="Phantom right_click via AXPress")
                self.logger.info("Phantom right_click failed, falling back to physical HID")
            elif action_type == "double_click":
                first = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if first.success:
                    second = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                    if second.success:
                        return ActionResult(success=True, reason="Phantom double_click via AXPress")
                self.logger.info("Phantom double_click failed, falling back to physical HID")

        if action_type == "mouse_move" and x is not None and y is not None:
            return self.hid_driver.move(x, y)
        if action_type == "left_click" and x is not None and y is not None:
            return self.hid_driver.left_click(x, y)
        if action_type == "right_click" and x is not None and y is not None:
            return self.hid_driver.right_click(x, y)
        if action_type == "double_click" and x is not None and y is not None:
            return self.hid_driver.double_click(x, y)
        
        if action_type == "drag_and_drop":
            tx, ty = action.get("target_x"), action.get("target_y")
            if x is not None and y is not None and tx is not None and ty is not None:
                return self.hid_driver.drag_and_drop(
                    x, y, tx, ty, 
                    duration=action.get("duration", 0.5),
                    hold_delay=action.get("hold_delay", 0.0)
                )
            return ActionResult(success=False, reason="drag_and_drop missing coordinates")
        
        if action_type == "select_area":
            tx, ty = action.get("target_x"), action.get("target_y")
            if x is not None and y is not None and tx is not None and ty is not None:
                return self.hid_driver.select_area(
                    x, y, tx, ty,
                    duration=action.get("duration", 0.4),
                    hold_delay=action.get("hold_delay", 0.0)
                )
            return ActionResult(success=False, reason="select_area missing coordinates")

        if action_type == "hover":
            if x is not None and y is not None:
                return self.hid_driver.hover(x, y, duration=action.get("duration", 1.0))
            return ActionResult(success=False, reason="hover missing coordinates")

        if action_type == "scroll":
            clicks = action.get("clicks", 0)
            axis = action.get("axis", "vertical")
            return self.hid_driver.scroll(int(clicks), axis=axis)
            
        if action_type == "type":
            return self.hid_driver.type_text(action.get("text", ""))
        if action_type == "key":
            keys = action.get("keys") or []
            return self.hid_driver.press_keys(keys)

        if action_type == "open_app":
            app_name = action.get("app_name", "")
            self.logger.info("Executing open_app sequence for: %s", app_name)
            
            # 1. Open Spotlight
            res = self.hid_driver.press_keys(["command", "space"])
            if not res.success:
                return res
            time.sleep(0.5) # Wait for Spotlight animation
            
            # 2. Type App Name
            res = self.hid_driver.type_text(app_name)
            if not res.success:
                return res
            time.sleep(0.3) # Wait for search results
            
            # 3. Press Enter
            return self.hid_driver.press_keys(["enter"])

        self.logger.warning("Unknown HID action: %s", action)
        return ActionResult(success=False, reason="unknown action")

    def _run_macro_actions(self, actions: list[dict]) -> ActionResult:
        """Execute a sequence of sub-actions in order. Stops on first failure."""
        subresults = []
        for idx, sub_action in enumerate(actions):
            # Prevent nested macro recursion
            if sub_action.get("type") == "macro_actions":
                subresults.append({"index": idx, "success": False, "reason": "nested macro not allowed"})
                return ActionResult(success=False, reason="nested macro not allowed", metadata={"subresults": subresults})

            res = self.execute(sub_action)
            subresults.append({"index": idx, "success": res.success, "reason": res.reason, "action": sub_action})
            if not res.success:
                return ActionResult(success=False, reason=f"macro step {idx} failed: {res.reason}", metadata={"subresults": subresults})

        return ActionResult(success=True, reason="macro complete", metadata={"subresults": subresults})
