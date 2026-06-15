from __future__ import annotations

import json
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
from macos_cua_agent.drivers.accessibility_driver import AccessibilityDriver
from macos_cua_agent.drivers.browser_driver import BrowserDriver
from macos_cua_agent.drivers.hid_driver import HIDDriver
from macos_cua_agent.drivers.semantic_driver import SemanticDriver
from macos_cua_agent.drivers.shell_driver import ShellDriver
from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.ax_utils import flatten_nodes_with_frames
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info


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
        self.shell_driver: BaseShellDriver = ShellDriver(settings)
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
        hitl_reason = (decision.reason or "policy hitl") if decision.hitl_required else ""
        if not hitl_reason and action.get("requires_hitl_confirmation"):
            hitl_reason = str(action.get("hitl_reason") or "autonomy policy confirmation")
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
        if action.get("type") == "wait_for_idle":
            return self._wait_for_idle(action)

        self.logger.info("Executing action via %s: %s", action.get("execution", "hid"), action)
        execution_path = action.get("execution", "hid")
        action_type = action.get("type")

        if action_type == "click_element":
            return self._click_element(action)
        if action_type == "fill_field":
            return self._fill_field(action)
        if action_type == "wait_for_element":
            return self._wait_for_element(action)
        if action_type == "scroll_to_element":
            return self._scroll_to_element(action)
        if action_type == "focus_window":
            return self._focus_window(action)
        if action_type == "click_and_type":
            return self._click_and_type(action)

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
            return self.browser_driver.execute_browser_action(action)
        elif execution_path == "semantic" and self.settings.enable_semantic:
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

    def _enrich_action(self, action: dict) -> dict:
        """
        Adds contextual fields needed for safety checks (e.g., current URL for browser ops).
        """
        if action.get("execution") == "browser" and action.get("command") == "run_javascript":
            try:
                page_url = self.browser_driver.get_current_url(action.get("app_name", "Safari"))
                if page_url:
                    action = dict(action)
                    action["page_url"] = page_url
            except Exception:
                # Non-fatal: if we can't retrieve the URL, proceed without it
                pass
        return action

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

    def _click_element(self, action: dict) -> ActionResult:
        target, reason = self._resolve_semantic_target(action, allow_coordinate_fallback=True)
        if not target:
            return ActionResult(success=False, reason=reason or "click_element target not found")

        click_type = str(action.get("click_type") or "left").strip().lower()
        action_type = {
            "left": "left_click",
            "right": "right_click",
            "double": "double_click",
        }.get(click_type, "left_click")
        payload = {
            "type": action_type,
            "x": target["x"],
            "y": target["y"],
            "phantom_mode": bool(action.get("phantom_mode", True)),
            "semantic_role": target.get("semantic_role", ""),
            "semantic_label": target.get("semantic_label", ""),
            "semantic_path": target.get("semantic_path", ""),
        }
        result = self._execute_hid(payload)
        if result.success:
            metadata = dict(result.metadata or {})
            metadata["target"] = target
            return ActionResult(success=True, reason=result.reason or "clicked element", metadata=metadata)
        return result

    def _fill_field(self, action: dict) -> ActionResult:
        text = str(action.get("text") or "")
        if not text:
            return ActionResult(success=False, reason="fill_field requires text")

        target, reason = self._resolve_semantic_target(action, allow_coordinate_fallback=True)
        if not target:
            return ActionResult(success=False, reason=reason or "fill_field target not found")

        x = target["x"]
        y = target["y"]
        phantom_mode = bool(action.get("phantom_mode", True))
        clear_first = bool(action.get("clear", True))

        if phantom_mode:
            set_res = self.accessibility_driver.set_text_element_value(x, y, text)
            if set_res.success:
                metadata = dict(set_res.metadata or {})
                metadata["target"] = target
                return ActionResult(success=True, reason="filled field via accessibility API", metadata=metadata)

        focus_res = self._execute_hid(
            {
                "type": "left_click",
                "x": x,
                "y": y,
                "phantom_mode": phantom_mode,
                "semantic_role": target.get("semantic_role", ""),
                "semantic_label": target.get("semantic_label", ""),
                "semantic_path": target.get("semantic_path", ""),
            }
        )
        if not focus_res.success:
            return focus_res

        if clear_first:
            select_res = self.hid_driver.press_keys([self._primary_modifier_key(), "a"])
            if not select_res.success:
                return ActionResult(success=False, reason=f"fill_field failed to select all: {select_res.reason}")
            backspace_res = self.hid_driver.press_keys(["backspace"])
            if not backspace_res.success:
                return ActionResult(success=False, reason=f"fill_field failed to clear text: {backspace_res.reason}")

        type_res = self.hid_driver.type_text(text)
        if not type_res.success:
            return type_res
        return ActionResult(success=True, reason="filled field", metadata={"target": target})

    def _click_and_type(self, action: dict) -> ActionResult:
        text = str(action.get("text") or "")
        if not text:
            return ActionResult(success=False, reason="click_and_type requires text")

        target, reason = self._resolve_semantic_target(action, allow_coordinate_fallback=True)
        if not target:
            return ActionResult(success=False, reason=reason or "click_and_type target not found")

        click_res = self._execute_hid(
            {
                "type": "left_click",
                "x": target["x"],
                "y": target["y"],
                "phantom_mode": bool(action.get("phantom_mode", True)),
                "semantic_role": target.get("semantic_role", ""),
                "semantic_label": target.get("semantic_label", ""),
                "semantic_path": target.get("semantic_path", ""),
            }
        )
        if not click_res.success:
            return click_res

        if bool(action.get("clear", True)):
            select_res = self.hid_driver.press_keys([self._primary_modifier_key(), "a"])
            if not select_res.success:
                return ActionResult(success=False, reason=f"click_and_type failed to select all: {select_res.reason}")
            backspace_res = self.hid_driver.press_keys(["backspace"])
            if not backspace_res.success:
                return ActionResult(success=False, reason=f"click_and_type failed to clear text: {backspace_res.reason}")

        type_res = self.hid_driver.type_text(text)
        if not type_res.success:
            return type_res

        if bool(action.get("submit", True)):
            submit_res = self.hid_driver.press_keys(["enter"])
            if not submit_res.success:
                return ActionResult(success=False, reason=f"click_and_type failed to submit: {submit_res.reason}")

        return ActionResult(success=True, reason="click_and_type complete", metadata={"target": target})

    def _wait_for_element(self, action: dict) -> ActionResult:
        timeout = max(0.1, float(action.get("timeout", 10.0)))
        deadline = time.time() + timeout
        last_reason = "element not found"

        while time.time() <= deadline:
            target, reason = self._resolve_semantic_target(action, allow_coordinate_fallback=False)
            if target:
                return ActionResult(success=True, reason="element found", metadata={"target": target})
            if reason:
                last_reason = reason
            time.sleep(0.2)

        return ActionResult(success=False, reason=f"wait_for_element timeout: {last_reason}")

    def _wait_for_idle(self, action: dict) -> ActionResult:
        timeout = max(0.1, float(action.get("timeout", 10.0)))
        deadline = time.time() + timeout
        stable_required = 3
        stable_count = 0

        prev_frame = None
        if self.vision_pipeline:
            try:
                prev_frame = self.vision_pipeline.capture_base64()
            except Exception:
                prev_frame = None
        prev_ax_sig = self._capture_ax_signature()

        if prev_frame is None and prev_ax_sig is None:
            time.sleep(min(0.3, timeout))
            return ActionResult(success=True, reason="wait_for_idle skipped (no sensors)")

        while time.time() <= deadline:
            time.sleep(0.2)
            changed = False

            if self.vision_pipeline and prev_frame is not None:
                try:
                    frame = self.vision_pipeline.capture_base64()
                    if self.vision_pipeline.has_changed(prev_frame, frame, threshold=0.002):
                        changed = True
                    prev_frame = frame
                except Exception:
                    pass

            ax_sig = self._capture_ax_signature()
            if prev_ax_sig is not None and ax_sig is not None and ax_sig != prev_ax_sig:
                changed = True
            if ax_sig is not None:
                prev_ax_sig = ax_sig

            if changed:
                stable_count = 0
            else:
                stable_count += 1
                if stable_count >= stable_required:
                    return ActionResult(success=True, reason="ui reached idle state")

        return ActionResult(success=False, reason="wait_for_idle timeout")

    def _scroll_to_element(self, action: dict) -> ActionResult:
        max_scrolls = max(1, int(action.get("max_scrolls", 24)))
        timeout = max(0.1, float(action.get("timeout", 10.0)))
        clicks = int(action.get("clicks", -8))
        axis = str(action.get("axis", "vertical") or "vertical")
        deadline = time.time() + timeout
        last_reason = "element not found"

        attempts = 0
        while time.time() <= deadline:
            target, reason = self._resolve_semantic_target(action, allow_coordinate_fallback=False)
            if target:
                return ActionResult(
                    success=True,
                    reason="element is visible",
                    metadata={"target": target, "scroll_attempts": attempts},
                )
            if reason:
                last_reason = reason
            if attempts >= max_scrolls:
                break

            scroll_res = self.hid_driver.scroll(clicks, axis=axis)
            if not scroll_res.success:
                return ActionResult(success=False, reason=f"scroll_to_element failed: {scroll_res.reason}")
            attempts += 1
            time.sleep(0.12)

        return ActionResult(success=False, reason=f"scroll_to_element timeout: {last_reason}")

    def _focus_window(self, action: dict) -> ActionResult:
        title = str(action.get("window_title") or "").strip()
        if not title:
            return ActionResult(success=False, reason="focus_window requires window_title")

        if self.settings.enable_semantic:
            res = self.semantic_driver.execute({"command": "focus_window", "window_title": title})
            if res.success:
                return res
            fallback = self.semantic_driver.execute({"command": "focus_app", "app_name": title})
            if fallback.success:
                return fallback
            return ActionResult(success=False, reason=f"focus_window failed: {res.reason}")

        return ActionResult(success=False, reason="semantic driver disabled for focus_window")

    def _resolve_semantic_target(
        self,
        action: dict,
        *,
        allow_coordinate_fallback: bool,
        max_depth: int = 5,
    ) -> tuple[dict | None, str]:
        anchor_x = self._as_float(action.get("x"))
        anchor_y = self._as_float(action.get("y"))
        hint_role = (action.get("semantic_role") or action.get("role") or "").strip()
        hint_label = (action.get("semantic_label") or action.get("label") or "").strip()
        hint_path = (action.get("semantic_path") or action.get("path") or "").strip()
        element_ref = str(action.get("element_ref") or action.get("element_id") or "").strip()

        ax_res = self.accessibility_driver.get_active_window_tree(max_depth=max_depth)
        if ax_res.success:
            ax_tree = (ax_res.metadata or {}).get("tree")
            if ax_tree:
                nodes = flatten_nodes_with_frames(ax_tree, max_nodes=200)
                if nodes:
                    best = self._pick_semantic_node(
                        nodes,
                        hint_role,
                        hint_label,
                        hint_path,
                        anchor_x,
                        anchor_y,
                        element_ref=element_ref,
                    )
                    if best:
                        frame = best.get("frame") or {}
                        try:
                            cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
                            cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
                            return {
                                "x": cx,
                                "y": cy,
                                "semantic_role": best.get("role", ""),
                                "semantic_label": best.get("label", ""),
                                "semantic_path": best.get("path", ""),
                            }, ""
                        except Exception:
                            pass
                reason = "semantic target not found in accessibility tree"
            else:
                reason = "accessibility tree missing"
        else:
            reason = ax_res.reason or "accessibility tree unavailable"

        if self.vision_pipeline:
            try:
                frame = self.vision_pipeline.capture_base64()
                visual_nodes = self.vision_pipeline.detect_ui_elements(frame)
                best = self._pick_visual_node(
                    visual_nodes,
                    hint_role,
                    hint_label or element_ref,
                    hint_path,
                    anchor_x,
                    anchor_y,
                    allow_anchor_only=allow_coordinate_fallback,
                )
                if best:
                    frame_data = best.get("frame") or {}
                    cx = float(frame_data.get("x", 0)) + float(frame_data.get("w", 0)) / 2.0
                    cy = float(frame_data.get("y", 0)) + float(frame_data.get("h", 0)) / 2.0
                    return {
                        "x": cx,
                        "y": cy,
                        "semantic_role": best.get("role", ""),
                        "semantic_label": best.get("label", best.get("title", "")),
                        "semantic_path": best.get("path", ""),
                        "grounding_source": best.get("source", "vision"),
                    }, "resolved via visual fallback"
            except Exception as exc:
                self.logger.debug("visual target resolution failed: %s", exc)

        if allow_coordinate_fallback and anchor_x is not None and anchor_y is not None:
            return {
                "x": anchor_x,
                "y": anchor_y,
                "semantic_role": hint_role,
                "semantic_label": hint_label or element_ref,
                "semantic_path": hint_path,
            }, reason
        return None, reason

    def _capture_ax_signature(self) -> str | None:
        if not self.settings.enable_semantic:
            return None
        ax_res = self.accessibility_driver.get_active_window_tree(max_depth=4)
        if not ax_res.success:
            return None
        tree = (ax_res.metadata or {}).get("tree")
        if not tree:
            return None
        try:
            return json.dumps(tree, sort_keys=True, ensure_ascii=False)
        except Exception:
            return None

    def _primary_modifier_key(self) -> str:
        return "command"

    def _as_float(self, value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _handle_clipboard(self, action: dict) -> ActionResult:
        sub = action.get("sub_action")
        try:
            if sub == "read":
                if action.get("capture_selection"):
                    select_res = self.hid_driver.press_keys([self._primary_modifier_key(), "a"])
                    if not select_res.success:
                        return ActionResult(success=False, reason=f"clipboard capture failed: {select_res.reason}")
                    copy_res = self.hid_driver.press_keys([self._primary_modifier_key(), "c"])
                    if not copy_res.success:
                        return ActionResult(success=False, reason=f"clipboard capture failed: {copy_res.reason}")
                    time.sleep(0.08)
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
                if action.get("paste"):
                    paste_res = self.hid_driver.press_keys([self._primary_modifier_key(), "v"])
                    if not paste_res.success:
                        return ActionResult(success=False, reason=f"clipboard paste failed: {paste_res.reason}")
                    return ActionResult(success=True, reason="wrote to clipboard and pasted")
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
        action = dict(action)
        action_type = action.get("type")
        x = action.get("x")
        y = action.get("y")
        phantom_mode = action.get("phantom_mode")
        # Default to phantom when we have semantic grounding (element_id) unless explicitly disabled
        if phantom_mode is None and action.get("element_id") is not None:
            phantom_mode = True
        phantom_mode = bool(phantom_mode)
        phantom_failed = False

        # Phantom Mode Check for Clicks
        if phantom_mode and action_type in ("left_click", "right_click", "double_click") and x is not None and y is not None:
            self.logger.info("Attempting Phantom Mode action for %s at (%s, %s)", action_type, x, y)
            if action_type == "left_click":
                res = self.accessibility_driver.perform_action_at(x, y, "AXPress")
                if res.success:
                    return res
            elif action_type == "right_click":
                # Try context-menu invocation if available; fall back to AXPress.
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
            time.sleep(0.5) # Wait for Spotlight animation
            
            # 3. Type App Name
            res = self.hid_driver.type_text(app_name)
            if not res.success:
                return res
            time.sleep(0.3) # Wait for search results
            
            # 4. Press Enter
            return self.hid_driver.press_keys(["enter"])

        self.logger.warning("Unknown HID action: %s", action)
        return ActionResult(success=False, reason="unknown action")

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

    def _run_macro_actions(self, actions: list[dict]) -> ActionResult:
        """Execute a sequence of sub-actions in order. Stops on first failure."""
        subresults = []
        for idx, sub_action in enumerate(actions):
            # Prevent nested macro recursion
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
                return ActionResult(success=False, reason=f"macro step {idx} failed: {res.reason}", metadata={"subresults": subresults})

        return ActionResult(success=True, reason="macro complete", metadata={"subresults": subresults})

    def _retarget_action_semantically(self, action: dict) -> tuple[dict, bool]:
        """
        Best-effort re-localization of a recorded action using semantic hints (role/label) and the current AX tree.
        This lets skills adapt when coordinates or overlay IDs from the recording are stale.
        """
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
        ax_tree = ax_res.metadata.get("tree")
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
        *,
        element_ref: str = "",
    ) -> dict | None:
        """
        Score nodes by semantic closeness (role/label) and optionally distance to prior coordinates.
        Returns the best-matching node or None if nothing reasonable is found.
        """
        best = None
        best_score = 0.0
        anchor = (anchor_x, anchor_y) if anchor_x is not None and anchor_y is not None else None
        hint_role_l = hint_role.lower()
        hint_label_l = hint_label.lower()
        hint_path_l = hint_path.lower()
        element_ref_l = str(element_ref or "").strip().lower()

        ref_tokens = [token for token in re.split(r"[^a-z0-9]+", element_ref_l) if len(token) > 1]
        generic_ref_tokens = {
            "btn",
            "button",
            "input",
            "field",
            "text",
            "textbox",
            "link",
            "label",
            "item",
            "icon",
            "element",
            "window",
            "pane",
            "tab",
            "menu",
            "msg",
            "message",
        }
        ref_tokens = [token for token in ref_tokens if token not in generic_ref_tokens]
        element_ref_norm = re.sub(r"[^a-z0-9]+", "", element_ref_l)

        for node in nodes:
            role = (node.get("role") or "").lower()
            label = (node.get("label") or "").lower()
            node_path = (node.get("path") or "").lower()
            combined = f"{role} {label} {node_path}".strip()
            combined_norm = re.sub(r"[^a-z0-9]+", "", combined)
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

            if element_ref_l:
                if element_ref_l in {role, label, node_path}:
                    score += 6.0
                if element_ref_l in label:
                    score += 3.0
                if element_ref_l in node_path:
                    score += 2.5
                if element_ref_l in role:
                    score += 1.5
                if element_ref_norm and element_ref_norm in combined_norm:
                    score += 2.5
                if ref_tokens:
                    matched = 0
                    for token in ref_tokens:
                        if token in combined_norm:
                            score += 1.2
                            matched += 1
                    if matched == len(ref_tokens):
                        score += 2.0

            if score <= 0:
                continue

            if anchor:
                frame = node.get("frame") or {}
                cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
                cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
                dx = cx - anchor[0]
                dy = cy - anchor[1]
                # Small tie-breaker favoring proximity to the original coordinate
                score -= math.hypot(dx, dy) * 0.001

            if score > best_score:
                best_score = score
                best = node

        return best if best_score > 0 else None
