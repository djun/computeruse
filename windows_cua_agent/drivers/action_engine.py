from __future__ import annotations

import math
import re
import subprocess
import sys
import time

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.drivers import (
    BaseAccessibilityDriver,
    BaseBrowserDriver,
    BaseHIDDriver,
    BaseSemanticDriver,
    BaseShellDriver,
    BaseVisionPipeline,
)
from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.ax_utils import flatten_nodes_with_frames
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


class ActionEngine:
    """Routes actions through policy evaluation, semantic path, or HID injection."""

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
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute(self, action: dict) -> ActionResult:
        action = self._enrich_action(action)
        allowed, deny_reason = self._is_allowed_by_execution_profile(action)
        if not allowed:
            return ActionResult(success=False, reason=deny_reason)

        decision = self.policy_engine.evaluate(action)
        if not decision.allowed:
            self.logger.warning("Action blocked by policy: %s", decision.reason)
            return ActionResult(success=False, reason=decision.reason)

        hitl_reason = decision.reason if decision.hitl_required else ""
        if not hitl_reason and self._requires_hitl(action):
            hitl_reason = "windows high-risk heuristic"

        if hitl_reason:
            self.logger.warning("Action requires human confirmation: %s", action)
            approved, deny_reason = self._request_hitl_approval(action, hitl_reason)
            if not approved:
                return ActionResult(success=False, reason=deny_reason)

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
            ax_res = self.accessibility_driver.get_active_window_tree()
            if ax_res.success:
                return ax_res
            return self._inspect_ui_visual_fallback(ax_res.reason)

        if action_type == "probe_ui":
            x = action.get("x")
            y = action.get("y")
            if x is None or y is None:
                return ActionResult(success=False, reason="probe_ui requires x,y coordinates")
            radius = float(action.get("radius") or 0.0)
            ax_probe = self.accessibility_driver.probe_element(x, y, radius=radius)
            if ax_probe.success:
                return ax_probe
            visual_probe = self._probe_ui_visual_fallback(float(x), float(y), ax_probe.reason)
            return visual_probe if visual_probe.success else ax_probe

        if action_type == "clipboard_op":
            return self._handle_clipboard(action)

        if action_type == "macro_actions":
            return self._run_macro_actions(action.get("actions") or [])

        if execution_path == "browser":
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
        if execution_path == "semantic" and self.settings.enable_semantic:
            result = self.semantic_driver.execute(action)
        elif execution_path == "shell":
            result = self.shell_driver.execute(action)
        else:
            result = self._execute_hid(action)

        self.logger.info("Action result: success=%s reason=%s", result.success, result.reason)
        return result

    def _is_allowed_by_execution_profile(self, action: dict) -> tuple[bool, str]:
        profile = self.settings.execution_profile
        execution_path = str(action.get("execution") or "hid")
        action_type = str(action.get("type") or "")

        if profile == "remote_cli":
            allowed_types = {"noop", "capture_only", "wait", "sandbox_shell", "script_op"}
            if action_type in allowed_types:
                if action_type in {"sandbox_shell", "script_op"} and execution_path != "shell":
                    return False, (
                        f"execution profile 'remote_cli' requires shell execution for {action_type}"
                    )
                return True, ""
            if execution_path == "shell":
                return True, ""
            return False, "execution profile 'remote_cli' blocks GUI/browser actions"

        if profile == "local_gui":
            if action_type in {"sandbox_shell", "script_op"} or execution_path == "shell":
                return False, "execution profile 'local_gui' blocks shell actions"

        return True, ""

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

    def _request_hitl_approval(self, action: dict, reason: str) -> tuple[bool, str]:
        if not self.settings.enable_hitl_prompt:
            self.logger.warning("HITL prompt disabled; denying high-risk action.")
            return False, "human confirmation required"

        stdin = getattr(sys, "stdin", None)
        if not stdin or not hasattr(stdin, "isatty") or not stdin.isatty():
            self.logger.warning("HITL confirmation required but no interactive stdin; denying action.")
            return False, "human confirmation required"

        summary = self._summarize_hitl_action(action)
        prompt = (
            "\n[HITL] High-risk action requested.\n"
            f"[HITL] Reason: {reason}\n"
            f"[HITL] Action: {summary}\n"
            "[HITL] Approve execution? [y/N]: "
        )
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "human confirmation required"

        if answer in {"y", "yes"}:
            return True, ""
        return False, "human confirmation denied"

    def _summarize_hitl_action(self, action: dict) -> str:
        action_type = str(action.get("type") or "unknown")
        if action_type == "sandbox_shell":
            command = str(action.get("cmd") or action.get("command") or "")
            return f"sandbox_shell cmd={command[:220]}"
        if action.get("execution") == "browser":
            command = str(action.get("command") or "")
            page_url = str(action.get("page_url") or action.get("url") or "")
            return f"browser command={command} url={page_url[:220]}"

        keys = [
            "type",
            "execution",
            "command",
            "operation",
            "path",
            "app_name",
            "bundle_id",
            "x",
            "y",
            "target_x",
            "target_y",
        ]
        details = {k: action.get(k) for k in keys if action.get(k) is not None}
        return str(details)

    def _enrich_action(self, action: dict) -> dict:
        """
        Adds contextual fields needed for safety checks (e.g., current URL for browser ops),
        and attaches the active process name to `bundle_id` for Windows policy rules.
        """
        enriched = dict(action)

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

        if enriched.get("execution") == "browser" and enriched.get("command") == "run_javascript":
            try:
                page_url = self.browser_driver.get_current_url(enriched.get("app_name", "Chrome"))
                if page_url:
                    enriched["page_url"] = page_url
            except Exception:
                pass

        return enriched

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

    def _inspect_ui_visual_fallback(self, ax_reason: str) -> ActionResult:
        if not self.vision_pipeline:
            return ActionResult(success=False, reason=ax_reason or "inspect_ui failed and vision fallback unavailable")

        try:
            frame = self.vision_pipeline.capture_base64()
            elements = self.vision_pipeline.detect_ui_elements(frame)
        except Exception as exc:
            return ActionResult(success=False, reason=f"visual inspect fallback failed: {exc}")

        if not elements:
            reason = "visual grounding found no elements"
            if ax_reason:
                reason = f"{ax_reason}; {reason}"
            return ActionResult(success=False, reason=reason)

        tree = {
            "role": "AXWindow",
            "title": "Visual Fallback",
            "frame": {"x": 0, "y": 0, "w": self.display.logical_width, "h": self.display.logical_height},
            "children": elements,
        }
        reason = "captured via visual grounding fallback"
        if ax_reason:
            reason = f"{reason} (after AX failure: {ax_reason})"
        return ActionResult(
            success=True,
            reason=reason,
            metadata={"tree": tree, "grounding": "vision_fallback", "element_count": len(elements)},
        )

    def _probe_ui_visual_fallback(self, x: float, y: float, ax_reason: str) -> ActionResult:
        if not self.vision_pipeline:
            return ActionResult(success=False, reason=ax_reason or "probe_ui failed and vision fallback unavailable")
        try:
            frame = self.vision_pipeline.capture_base64()
            elements = self.vision_pipeline.detect_ui_elements(frame)
        except Exception as exc:
            return ActionResult(success=False, reason=f"visual probe fallback failed: {exc}")

        if not elements:
            reason = "visual grounding found no elements"
            if ax_reason:
                reason = f"{ax_reason}; {reason}"
            return ActionResult(success=False, reason=reason)

        best = self._pick_visual_node(elements, "", "", "", x, y, allow_anchor_only=True)
        if not best:
            return ActionResult(success=False, reason=ax_reason or "visual probe fallback could not localize element")

        metadata = {"tree": best, "grounding": "vision_fallback"}
        reason = "probed via visual grounding fallback"
        if ax_reason:
            reason = f"{reason} (after AX failure: {ax_reason})"
        return ActionResult(success=True, reason=reason, metadata=metadata)

    def _handle_clipboard(self, action: dict) -> ActionResult:
        sub = action.get("sub_action")
        try:
            if sub == "read":
                content = subprocess.check_output(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"],
                    text=True,
                )
                sensitive, redacted = self._redact_clipboard_content(content)
                return ActionResult(
                    success=True,
                    reason="read clipboard (redacted)" if sensitive else "read clipboard",
                    metadata={"content": redacted, "redacted": sensitive},
                )
            if sub == "write":
                content = action.get("content", "")
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                    input=content,
                    text=True,
                    check=True,
                )
                return ActionResult(success=True, reason="wrote to clipboard")
            if sub == "clear":
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Set-Clipboard -Value ''"],
                    check=True,
                )
                return ActionResult(success=True, reason="cleared clipboard")
        except Exception as exc:
            return ActionResult(success=False, reason=f"clipboard op failed: {exc}")
        return ActionResult(success=False, reason=f"unknown clipboard sub_action: {sub}")

    def _redact_clipboard_content(self, content: str) -> tuple[bool, str]:
        """Lightweight secret detector to avoid leaking sensitive clipboard contents back to the model."""
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

        if len(content) >= 32 and self._shannon_entropy(content) > 4.0:
            return True, "<REDACTED>"

        return False, content

    def _shannon_entropy(self, data: str) -> float:
        freq = {ch: data.count(ch) for ch in set(data)}
        length = len(data) or 1
        return -sum((count / length) * math.log2(count / length) for count in freq.values())

    def _execute_hid(self, action: dict) -> ActionResult:
        action = dict(action)
        action_type = action.get("type")
        x = action.get("x")
        y = action.get("y")
        phantom_mode = action.get("phantom_mode")
        if phantom_mode is None and action.get("element_id") is not None:
            phantom_mode = True
        phantom_mode = bool(phantom_mode)
        phantom_failed = False

        # Phantom mode: attempt semantic click before physical HID.
        if phantom_mode and action_type in ("left_click", "right_click", "double_click") and x is not None and y is not None:
            self.logger.info("Attempting Phantom Mode action for %s at (%s, %s)", action_type, x, y)
            if action_type == "left_click":
                res = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if res.success:
                    return res
            elif action_type == "right_click":
                res = self.accessibility_driver.perform_action_at(x, y, "AXShowMenu")
                if res.success:
                    return res
                res = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if res.success:
                    return ActionResult(success=True, reason="Phantom right_click via AXPress")
            elif action_type == "double_click":
                first = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if first.success:
                    second = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                    if second.success:
                        return ActionResult(success=True, reason="Phantom double_click via AXPress")

            phantom_failed = True
            self.logger.info("Phantom mode failed, attempting visual grounding fallback")

        if phantom_failed:
            action, retargeted = self._retarget_action_visually(action, allow_anchor_only=True)
            if retargeted:
                x = action.get("x")
                y = action.get("y")
                self.logger.info(
                    "Visual fallback retargeted %s to (%s, %s) via %s",
                    action_type,
                    x,
                    y,
                    action.get("visual_source", "vision"),
                )

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
                    x, y, tx, ty, duration=action.get("duration", 0.5), hold_delay=action.get("hold_delay", 0.0)
                )
            return ActionResult(success=False, reason="drag_and_drop missing coordinates")

        if action_type == "select_area":
            tx, ty = action.get("target_x"), action.get("target_y")
            if x is not None and y is not None and tx is not None and ty is not None:
                return self.hid_driver.select_area(
                    x, y, tx, ty, duration=action.get("duration", 0.4), hold_delay=action.get("hold_delay", 0.0)
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
            text = action.get("text", "")
            if phantom_mode:
                if x is not None and y is not None:
                    res = self.accessibility_driver.set_text_element_value(x, y, text)
                    if res.success:
                        return res
                    self.logger.info("Phantom type at coordinates failed, trying focused element")
                focused_res = self.accessibility_driver.set_focused_element_value(text)
                if focused_res and focused_res.success:
                    return focused_res
                self.logger.info("Phantom type failed, attempting visual focus before physical HID typing")

            hint_role = (action.get("semantic_role") or action.get("role") or "").strip()
            hint_label = (action.get("semantic_label") or action.get("label") or "").strip()
            hint_path = (action.get("semantic_path") or action.get("path") or "").strip()
            if self.vision_pipeline and (phantom_mode or action.get("element_id") is not None or hint_role or hint_label or hint_path):
                focused_action, retargeted = self._retarget_action_visually(action, allow_anchor_only=x is not None and y is not None)
                if retargeted:
                    fx = focused_action.get("x")
                    fy = focused_action.get("y")
                    if fx is not None and fy is not None:
                        focus_res = self.hid_driver.left_click(fx, fy)
                        if focus_res.success:
                            time.sleep(0.05)
                        else:
                            self.logger.debug("Visual focus click failed before typing: %s", focus_res.reason)
            return self.hid_driver.type_text(text)

        if action_type == "key":
            keys = action.get("keys") or []
            return self.hid_driver.press_keys(keys)

        if action_type == "open_app":
            app_name = action.get("app_name", "")
            self.logger.info("Executing open_app for: %s", app_name)

            # Try semantic focus first (best-effort)
            if self.settings.enable_semantic:
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

        self.logger.warning("Unknown HID action: %s", action)
        return ActionResult(success=False, reason="unknown action")

    def _run_macro_actions(self, actions: list[dict]) -> ActionResult:
        subresults = []
        for idx, sub_action in enumerate(actions):
            if sub_action.get("type") == "macro_actions":
                subresults.append({"index": idx, "success": False, "reason": "nested macro not allowed"})
                return ActionResult(success=False, reason="nested macro not allowed", metadata={"subresults": subresults})

            adjusted_action = sub_action
            if self.settings.enable_semantic:
                adjusted_action, retargeted = self._retarget_action_semantically(sub_action)
                if retargeted:
                    self.logger.info("Retargeted macro action %s using semantic hints", sub_action.get("type"))

            res = self.execute(adjusted_action)
            subresults.append({"index": idx, "success": res.success, "reason": res.reason, "action": sub_action})
            if not res.success:
                return ActionResult(
                    success=False,
                    reason=f"macro step {idx} failed: {res.reason}",
                    metadata={"subresults": subresults},
                )

        return ActionResult(success=True, reason="macro complete", metadata={"subresults": subresults})

    def _retarget_action_semantically(self, action: dict) -> tuple[dict, bool]:
        pointer_actions = {
            "left_click",
            "right_click",
            "double_click",
            "hover",
            "mouse_move",
            "type",
            "drag_and_drop",
            "select_area",
        }
        if action.get("type") not in pointer_actions:
            return action, False

        hint_role = (action.get("semantic_role") or action.get("role") or "").strip()
        hint_label = (action.get("semantic_label") or action.get("label") or "").strip()
        hint_path = (action.get("semantic_path") or action.get("path") or "").strip()
        if not hint_role and not hint_label and not hint_path:
            return action, False

        ax_res = self.accessibility_driver.get_active_window_tree(max_depth=4)
        if not ax_res.success:
            return action, False
        ax_tree = ax_res.metadata.get("tree") if ax_res.metadata else None
        if not ax_tree:
            return action, False

        nodes = flatten_nodes_with_frames(ax_tree, max_nodes=80)
        if not nodes:
            return action, False

        best = self._pick_semantic_node(nodes, hint_role, hint_label, hint_path, action.get("x"), action.get("y"))
        if not best:
            return action, False

        frame = best.get("frame") or {}
        try:
            cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
            cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
        except Exception:
            return action, False

        updated = dict(action)
        updated["x"] = cx
        updated["y"] = cy
        updated.setdefault("semantic_role", best.get("role", ""))
        updated.setdefault("semantic_label", best.get("label", ""))
        return updated, True

    def _retarget_action_visually(self, action: dict, *, allow_anchor_only: bool = False) -> tuple[dict, bool]:
        pointer_actions = {
            "left_click",
            "right_click",
            "double_click",
            "hover",
            "mouse_move",
            "type",
        }
        if action.get("type") not in pointer_actions:
            return action, False
        if not self.vision_pipeline:
            return action, False

        hint_role = (action.get("semantic_role") or action.get("role") or "").strip()
        hint_label = (action.get("semantic_label") or action.get("label") or action.get("title") or "").strip()
        hint_path = (action.get("semantic_path") or action.get("path") or "").strip()
        anchor_x = action.get("x")
        anchor_y = action.get("y")
        has_hints = bool(hint_role or hint_label or hint_path)
        if not has_hints and not allow_anchor_only:
            return action, False
        if not has_hints and (anchor_x is None or anchor_y is None):
            return action, False

        try:
            frame = self.vision_pipeline.capture_base64()
            nodes = self.vision_pipeline.detect_ui_elements(frame)
        except Exception as exc:
            self.logger.debug("Visual retarget capture/detect failed: %s", exc)
            return action, False
        if not nodes:
            return action, False

        best = self._pick_visual_node(
            nodes,
            hint_role,
            hint_label,
            hint_path,
            float(anchor_x) if anchor_x is not None else None,
            float(anchor_y) if anchor_y is not None else None,
            allow_anchor_only=allow_anchor_only,
        )
        if not best:
            return action, False

        frame = best.get("frame") or {}
        try:
            cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
            cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
        except Exception:
            return action, False

        updated = dict(action)
        updated["x"] = cx
        updated["y"] = cy
        updated["visual_source"] = best.get("source", "vision")
        updated.setdefault("semantic_role", best.get("role", ""))
        updated.setdefault("semantic_label", best.get("label", best.get("title", "")))
        updated.setdefault("semantic_path", best.get("path", ""))
        return updated, True

    def _pick_visual_node(
        self,
        nodes: list[dict],
        hint_role: str,
        hint_label: str,
        hint_path: str,
        anchor_x: float | None,
        anchor_y: float | None,
        *,
        allow_anchor_only: bool,
    ) -> dict | None:
        best = None
        best_score = float("-inf")
        hint_role_l = hint_role.lower()
        hint_label_l = hint_label.lower()
        hint_path_l = hint_path.lower()
        anchor = (anchor_x, anchor_y) if anchor_x is not None and anchor_y is not None else None
        has_hints = bool(hint_role_l or hint_label_l or hint_path_l)

        for node in nodes:
            frame = node.get("frame") or {}
            try:
                cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
                cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
            except Exception:
                continue

            role = str(node.get("role") or "").lower()
            label = str(node.get("label") or node.get("title") or "").lower()
            path = str(node.get("path") or "").lower()
            src = str(node.get("source") or "").lower()
            conf = float(node.get("confidence", 0.0) or 0.0)

            if "detector" in src:
                source_bias = 1.2
            elif src == "ocr":
                source_bias = 0.8
            else:
                source_bias = 0.2

            if has_hints:
                score = source_bias + conf
                if hint_role_l:
                    if role == hint_role_l:
                        score += 3.0
                    elif hint_role_l in role:
                        score += 1.5
                if hint_label_l:
                    if label == hint_label_l:
                        score += 5.0
                    elif hint_label_l in label:
                        score += 2.5
                if hint_path_l:
                    if path == hint_path_l:
                        score += 4.0
                    elif hint_path_l in path:
                        score += 2.0
                if score <= source_bias + conf:
                    continue
                if anchor:
                    score -= math.hypot(cx - anchor[0], cy - anchor[1]) * 0.001
            else:
                if not allow_anchor_only or not anchor:
                    continue
                dist = math.hypot(cx - anchor[0], cy - anchor[1])
                score = source_bias + conf + (1.0 / (1.0 + dist))

            if score > best_score:
                best_score = score
                best = node

        return best

    def _pick_semantic_node(
        self,
        nodes: list[dict],
        hint_role: str,
        hint_label: str,
        hint_path: str,
        anchor_x: float | None,
        anchor_y: float | None,
    ) -> dict | None:
        best = None
        best_score = 0.0
        anchor = (anchor_x, anchor_y) if anchor_x is not None and anchor_y is not None else None
        hint_role_l = hint_role.lower()
        hint_label_l = hint_label.lower()
        hint_path_l = hint_path.lower()

        for node in nodes:
            role = (node.get("role") or "").lower()
            label = (node.get("label") or "").lower()
            node_path = (node.get("path") or "").lower()
            score = 0.0

            if hint_role_l:
                if role == hint_role_l:
                    score += 3.0
                elif hint_role_l in role:
                    score += 1.5

            if hint_label_l:
                if label == hint_label_l:
                    score += 4.0
                elif hint_label_l in label:
                    score += 2.0

            if hint_path_l:
                if node_path == hint_path_l:
                    score += 5.0
                elif hint_path_l in node_path:
                    score += 3.0

            if score <= 0:
                continue

            if anchor:
                frame = node.get("frame") or {}
                cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
                cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
                dx = cx - anchor[0]
                dy = cy - anchor[1]
                score -= math.hypot(dx, dy) * 0.001

            if score > best_score:
                best_score = score
                best = node

        return best if best_score > 0 else None
