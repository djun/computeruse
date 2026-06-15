"""Post-action verification service."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:  # pragma: no cover - optional dependency
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None

from cua_agent.agent.state_manager import StateManager, VerificationContract
from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.orchestrator.planning import Step
from cua_agent.utils.config import Settings


class VerificationManager:
    """Runs verification contracts using telemetry, a11y, and visual signals."""

    def __init__(self, settings: Settings, computer: ComputerAdapter) -> None:
        self.settings = settings
        self.computer = computer

    def resolve_contract(
        self,
        state: StateManager,
        action: dict[str, Any],
        current_step: Step | None,
    ) -> VerificationContract:
        fallback_expected = None
        if current_step and getattr(current_step, "expected_state", ""):
            fallback_expected = str(current_step.expected_state).strip() or None
        if fallback_expected is None:
            fallback_expected = self.default_expected_state_for_action(action)
        return state.normalize_verification_contract(
            action.get("verification") if isinstance(action.get("verification"), dict) else None,
            fallback_sensor=self.default_sensor_for_action(action, current_step),
            fallback_expected_state=fallback_expected,
            verify_after=action.get("verify_after"),
        )

    def default_sensor_for_action(self, action: dict[str, Any], current_step: Step | None = None) -> str:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type in {
            "wait",
            "wait_for_element",
            "wait_for_idle",
            "capture_only",
            "noop",
            "inspect_ui",
            "probe_ui",
            "notebook_op",
            "mouse_move",
            "hover",
            "scroll",
            "scroll_to_element",
        }:
            return "none"
        if action_type in {"sandbox_shell", "script_op"}:
            return "none"
        if action_type == "clipboard_op":
            sub_action = str(action.get("sub_action") or "").strip().lower()
            return "none" if sub_action == "read" else "os_telemetry"
        if action_type in {"open_app", "focus_window"}:
            return "os_telemetry"
        if action_type in {"browser_op"}:
            return "pixel_diff"
        # Generic action: let the step's preferred sensor override the a11y_tree
        # default. Action-specific defaults above take precedence so cheaper/safer
        # sensors (none/os_telemetry/pixel_diff) are not clobbered by the step's
        # default preferred_sensor of a11y_tree.
        if current_step:
            preferred = str(getattr(current_step, "preferred_sensor", "") or "").strip().lower()
            if preferred in StateManager.SENSOR_HIERARCHY and preferred != "none":
                return preferred
        return "a11y_tree"

    def default_expected_state_for_action(self, action: dict[str, Any]) -> str | None:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type != "clipboard_op":
            return None
        sub_action = str(action.get("sub_action") or "").strip().lower()
        if sub_action == "write":
            content = str(action.get("content") or "")
            if not content:
                return "clipboard_equals:"
            if len(content) > 220:
                return f"clipboard_contains:{content[:220]}"
            return f"clipboard_equals:{content}"
        if sub_action == "clear":
            return "clipboard_equals:"
        return None

    def collect_os_telemetry_snapshot(self, contract: VerificationContract) -> dict[str, Any]:
        if contract.sensor != "os_telemetry":
            return {}
        snapshot: dict[str, Any] = {}
        clipboard = self._read_clipboard_snapshot()
        if clipboard is not None:
            snapshot["clipboard"] = clipboard
        return snapshot

    def run_verification_contract(
        self,
        *,
        action: dict[str, Any],
        contract: VerificationContract,
        current_frame: str,
        current_hash: str | None,
        ax_tree_before: dict[str, Any] | None,
        telemetry_before: dict[str, Any],
        global_hotkeys: set[tuple[str, ...]],
        visual_hash_static_threshold: int,
    ) -> dict[str, Any]:
        sensor = contract.sensor

        if sensor == "none":
            next_frame, next_hash = self.computer.capture_with_hash()
            return {
                "passed": True,
                "reason": "verification bypassed by contract",
                "sensor": sensor,
                "changed": True,
                "next_frame": next_frame,
                "next_hash": next_hash,
                "hash_distance": self.computer.hash_distance(current_hash, next_hash),
                "ssim_score": None,
                "ax_tree_after": None,
                "ax_changed": False,
                "note": "verification:none",
                "force_vision_next_turn": False,
            }

        if sensor == "os_telemetry":
            passed, reason = self.verify_os_telemetry(contract, telemetry_before)
            if passed:
                next_frame, next_hash = self.computer.capture_with_hash()
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": next_frame,
                    "next_hash": next_hash,
                    "hash_distance": self.computer.hash_distance(current_hash, next_hash),
                    "ssim_score": None,
                    "ax_tree_after": None,
                    "ax_changed": False,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }
            visual = self.run_visual_verification(
                action=action,
                contract=contract,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                visual_hash_static_threshold=visual_hash_static_threshold,
            )
            if self.is_os_telemetry_inconclusive_reason(reason):
                visual_changed = bool(visual.get("changed"))
                visual["passed"] = visual_changed
                visual["reason"] = (
                    f"{reason}; visual fallback detected change"
                    if visual_changed
                    else f"{reason}; visual fallback found no change"
                )
                visual["note"] = f"verification:{sensor}:fallback"
            else:
                visual["passed"] = False
                visual["reason"] = reason
                visual["note"] = f"verification:{sensor}:timeout"
            visual["sensor"] = sensor
            visual["force_vision_next_turn"] = True
            return visual

        if sensor == "a11y_tree":
            passed, reason, ax_tree_after = self.verify_a11y_tree(contract, ax_tree_before)
            if passed:
                next_frame, next_hash = self.computer.capture_with_hash()
                ax_changed = self.ax_changed(ax_tree_before, ax_tree_after) if ax_tree_after else False
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": next_frame,
                    "next_hash": next_hash,
                    "hash_distance": self.computer.hash_distance(current_hash, next_hash),
                    "ssim_score": None,
                    "ax_tree_after": ax_tree_after,
                    "ax_changed": ax_changed,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }
            visual = self.run_visual_verification(
                action=action,
                contract=contract,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                visual_hash_static_threshold=visual_hash_static_threshold,
            )
            if self.is_a11y_unavailable_reason(reason):
                visual_changed = bool(visual.get("changed"))
                visual["passed"] = visual_changed
                visual["reason"] = (
                    f"{reason}; visual fallback detected change"
                    if visual_changed
                    else f"{reason}; visual fallback found no change"
                )
                visual["note"] = f"verification:{sensor}:fallback"
            else:
                visual["passed"] = False
                visual["reason"] = reason
                visual["note"] = f"verification:{sensor}:timeout"
            visual["sensor"] = sensor
            visual["force_vision_next_turn"] = True
            return visual

        if sensor == "pixel_diff":
            passed, reason, frame_after = self.verify_pixel_diff(contract, current_frame)
            if passed:
                next_hash = self.computer.hash_base64(frame_after)
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": frame_after,
                    "next_hash": next_hash,
                    "hash_distance": self.computer.hash_distance(current_hash, next_hash),
                    "ssim_score": None,
                    "ax_tree_after": None,
                    "ax_changed": False,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }
            visual = self.run_visual_verification(
                action=action,
                contract=contract,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                visual_hash_static_threshold=visual_hash_static_threshold,
            )
            visual["passed"] = False
            visual["reason"] = reason
            visual["sensor"] = sensor
            visual["note"] = f"verification:{sensor}:timeout"
            visual["force_vision_next_turn"] = True
            return visual

        visual = self.run_visual_verification(
            action=action,
            contract=contract,
            current_frame=current_frame,
            current_hash=current_hash,
            ax_tree_before=ax_tree_before,
            global_hotkeys=global_hotkeys,
            visual_hash_static_threshold=visual_hash_static_threshold,
        )
        visual["note"] = f"verification:{sensor}"
        visual["sensor"] = sensor
        visual["force_vision_next_turn"] = True
        return visual

    def run_visual_verification(
        self,
        *,
        action: dict[str, Any],
        contract: VerificationContract,
        current_frame: str,
        current_hash: str | None,
        ax_tree_before: dict[str, Any] | None,
        global_hotkeys: set[tuple[str, ...]],
        visual_hash_static_threshold: int,
    ) -> dict[str, Any]:
        verify_after = bool(action.get("verify_after", True))
        is_interactive = action.get("type") not in {"wait", "capture_only", "noop"}
        extra_delay = 0.0
        if verify_after and action.get("type") == "key":
            combo = tuple(sorted([str(k).lower() for k in action.get("keys") or []]))
            if combo in global_hotkeys:
                extra_delay = 0.5

        if verify_after and is_interactive:
            time.sleep(0.2 + extra_delay)
            stabilize_timeout = max(2.0, self.settings.settle_delay_ms / 1000.0)
            start_time = time.time()
            last_poll_frame = self.computer.capture_base64()
            stable_frames = 0
            while (time.time() - start_time) < stabilize_timeout:
                time.sleep(0.15)
                current_poll_frame = self.computer.capture_base64()
                if not self.computer.has_changed(last_poll_frame, current_poll_frame, threshold=0.002):
                    stable_frames += 1
                else:
                    stable_frames = 0
                last_poll_frame = current_poll_frame
                if stable_frames >= 2:
                    break
            next_frame = last_poll_frame
            next_hash = self.computer.hash_base64(next_frame)
        else:
            next_frame, next_hash = self.computer.capture_with_hash()

        hash_distance = self.computer.hash_distance(current_hash, next_hash)
        ssim_score = None
        ax_tree_after = None
        ax_changed = False

        if verify_after:
            ssim_score = self.computer.structural_similarity(current_frame, next_frame)
            if self.settings.enable_semantic:
                ax_after_res = self.computer.get_active_window_tree(max_depth=4)
                if ax_after_res.success:
                    ax_tree_after = ax_after_res.metadata.get("tree")
                    ax_changed = self.ax_changed(ax_tree_before, ax_tree_after)
            changed = self.compute_changed(
                current_frame,
                next_frame,
                hash_distance,
                ssim_score,
                ax_changed,
                visual_hash_static_threshold=visual_hash_static_threshold,
            )
            note = "verification:visual"
        else:
            changed = True
            note = "verify_skipped"

        passed = bool(changed)
        reason = "visual changed" if changed else "visual unchanged"
        expected_key, expected_value = self.parse_expected_state(contract.expected_state)
        if expected_key in {"target_region_changed", "target_region_stable"}:
            region_changed, region_reason = self.target_region_changed(
                current_frame,
                next_frame,
                action.get("target_frame") if isinstance(action.get("target_frame"), dict) else None,
            )
            passed = region_changed if expected_key == "target_region_changed" else not region_changed
            reason = region_reason
            changed = region_changed
        elif expected_key in {"visual_text_exists", "visual_contains"}:
            matched = self.visual_text_exists(next_frame, expected_value)
            passed = matched
            reason = "visual text found" if matched else "visual text not found"
        elif expected_key in {"visual_text_not_exists", "visual_not_contains"}:
            matched = not self.visual_text_exists(next_frame, expected_value)
            passed = matched
            reason = "visual text absent" if matched else "visual text still present"
        elif contract.expected_state:
            if expected_key in {"any", "state_change", "changed"}:
                passed = bool(changed)
                reason = "vision_full detected change" if changed else "vision_full found no change"
            else:
                matched, expected_reason = self.evaluate_a11y_state(
                    contract.expected_state, ax_tree_before, ax_tree_after
                )
                if matched:
                    passed = True
                    reason = expected_reason
                elif self.is_a11y_unavailable_reason(expected_reason):
                    passed = bool(changed)
                    reason = (
                        f"{expected_reason}; visual fallback detected change"
                        if changed
                        else f"{expected_reason}; visual fallback found no change"
                    )
                    note = "verification:vision_full:fallback"
                else:
                    passed = False
                    reason = expected_reason

        return {
            "passed": bool(passed),
            "reason": reason,
            "sensor": "vision_full",
            "changed": bool(changed),
            "next_frame": next_frame,
            "next_hash": next_hash,
            "hash_distance": hash_distance,
            "ssim_score": ssim_score,
            "ax_tree_after": ax_tree_after,
            "ax_changed": ax_changed,
            "note": note,
            "force_vision_next_turn": True,
        }

    def verify_os_telemetry(
        self, contract: VerificationContract, before_snapshot: dict[str, Any]
    ) -> tuple[bool, str]:
        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_reason = "telemetry condition unmet"
        while time.time() <= deadline:
            now_snapshot = self.collect_os_telemetry_snapshot(contract)
            ok, reason = self.evaluate_os_telemetry_state(contract.expected_state, before_snapshot, now_snapshot)
            if ok:
                return True, reason
            last_reason = reason
            if self.is_os_telemetry_inconclusive_reason(reason):
                return False, reason
            time.sleep(0.25)
        return False, last_reason

    def verify_a11y_tree(
        self, contract: VerificationContract, before_tree: dict[str, Any] | None
    ) -> tuple[bool, str, dict[str, Any] | None]:
        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_reason = "a11y condition unmet"
        last_tree: dict[str, Any] | None = None
        while time.time() <= deadline:
            ax_res = self.computer.get_active_window_tree(max_depth=4)
            if ax_res.success:
                last_tree = ax_res.metadata.get("tree")
                ok, reason = self.evaluate_a11y_state(contract.expected_state, before_tree, last_tree)
                if ok:
                    return True, reason, last_tree
                last_reason = reason
            else:
                last_reason = ax_res.reason or "a11y capture failed"
            time.sleep(0.35)
        return False, last_reason, last_tree

    def verify_pixel_diff(self, contract: VerificationContract, base_frame: str) -> tuple[bool, str, str]:
        key, value = self.parse_expected_state(contract.expected_state)
        threshold = 0.01
        if key in {"pixel_change_gt", "pixel_diff_gt", "change_gt"} and value:
            try:
                threshold = max(0.0005, min(float(value), 0.2))
            except ValueError:
                threshold = 0.01
        if key in {"pixel_change_pct_gt"} and value:
            try:
                threshold = max(0.0005, min(float(value) / 100.0, 0.2))
            except ValueError:
                threshold = 0.01

        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_frame = base_frame
        while time.time() <= deadline:
            frame = self.computer.capture_base64()
            last_frame = frame
            if self.computer.has_changed(base_frame, frame, threshold=threshold):
                return True, f"pixel delta exceeded threshold={threshold:.4f}", frame
            time.sleep(0.2)
        return False, f"pixel delta stayed below threshold={threshold:.4f}", last_frame

    def evaluate_os_telemetry_state(
        self,
        expected_state: str | None,
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
    ) -> tuple[bool, str]:
        key, value = self.parse_expected_state(expected_state)
        clipboard_before = str(before_snapshot.get("clipboard") or "")
        clipboard_after = str(after_snapshot.get("clipboard") or "")
        value_l = value.lower()

        if key in {"any", "state_change", "changed"}:
            changed = before_snapshot != after_snapshot
            if changed:
                if not self.has_non_clipboard_os_signal(before_snapshot, after_snapshot):
                    return False, "os telemetry inconclusive (no non-clipboard signal)"
                if not self.has_non_clipboard_os_delta(before_snapshot, after_snapshot):
                    return False, "os telemetry inconclusive (clipboard-only change)"
                return True, "os telemetry changed"
            if key in {"state_change", "changed"}:
                return False, "os telemetry unchanged"
            if not self.has_non_clipboard_os_signal(before_snapshot, after_snapshot):
                return False, "os telemetry inconclusive (no non-clipboard signal)"
            return False, "os telemetry unchanged"
        if key == "clipboard_changed":
            changed = clipboard_before != clipboard_after
            return changed, "clipboard changed" if changed else "clipboard unchanged"
        if key == "clipboard_contains":
            matched = value_l in clipboard_after.lower()
            return matched, "clipboard contains expected text" if matched else "clipboard missing expected text"
        if key == "clipboard_equals":
            matched = clipboard_after == value
            return matched, "clipboard equals expected text" if matched else "clipboard value mismatch"
        if key == "file_exists":
            exists = Path(value).expanduser().exists() if value else False
            return exists, "file exists" if exists else "file not found"
        if key in {"file_not_exists", "file_missing"}:
            missing = not Path(value).expanduser().exists() if value else True
            return missing, "file absent" if missing else "file still exists"
        if key in {"process_exists", "app_open"}:
            running = self.process_exists(value)
            return running, "process found" if running else "process not found"
        if key == "process_not_exists":
            stopped = not self.process_exists(value)
            return stopped, "process absent" if stopped else "process still running"
        if key == "app_focused":
            focused = self.process_exists(value) or (value_l in json.dumps(after_snapshot, ensure_ascii=False).lower())
            return focused, "app appears focused/open" if focused else "app focus not detected"

        if not self.has_non_clipboard_os_signal(before_snapshot, after_snapshot):
            return False, "os telemetry inconclusive (no non-clipboard signal)"
        blob = json.dumps(after_snapshot, ensure_ascii=False).lower()
        matched = key in blob if value == "" else value_l in blob
        return matched, "telemetry token found" if matched else "telemetry token missing"

    def evaluate_a11y_state(
        self,
        expected_state: str | None,
        before_tree: dict[str, Any] | None,
        after_tree: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if after_tree is None:
            return False, "a11y tree unavailable"
        key, value = self.parse_expected_state(expected_state)
        payload = json.dumps(after_tree, ensure_ascii=False).lower()
        value_l = value.lower()
        if key in {"any", "state_change", "changed"}:
            if before_tree is None:
                return True, "a11y tree captured"
            changed = self.ax_changed(before_tree, after_tree)
            return changed, "a11y tree changed" if changed else "a11y tree unchanged"
        if key in {"text_exists", "contains", "title_contains", "url_contains"}:
            matched = value_l in payload
            return matched, "a11y text found" if matched else "a11y text not found"
        if key in {"text_not_exists", "not_contains"}:
            matched = value_l not in payload
            return matched, "a11y text absent" if matched else "a11y text still present"
        if key == "role_exists":
            matched = f"\"role\": \"{value_l}\"" in payload or f"\"role\":\"{value_l}\"" in payload
            return matched, "a11y role found" if matched else "a11y role not found"
        token = key if value == "" else value_l
        matched = token in payload
        return matched, "a11y token found" if matched else "a11y token missing"

    def target_region_changed(
        self,
        before_b64: str,
        after_b64: str,
        target_frame: dict[str, Any] | None,
        padding: int = 12,
    ) -> tuple[bool, str]:
        if not target_frame:
            return False, "target frame unavailable"
        try:
            before = self._crop_region(self._decode(before_b64), target_frame, padding)
            after = self._crop_region(self._decode(after_b64), target_frame, padding)
        except Exception as exc:
            return False, f"target region decode failed: {exc}"
        if before.size != after.size or before.size[0] == 0 or before.size[1] == 0:
            return False, "target region invalid"
        changed = self._region_changed(before, after)
        return changed, "target region changed" if changed else "target region stable"

    def visual_text_exists(self, image_b64: str, text: str) -> bool:
        needle = str(text or "").strip().lower()
        if not needle:
            return False
        try:
            nodes = self.computer.detect_ui_elements(image_b64)
        except Exception:
            return False
        haystack = " ".join(
            str(node.get("label") or node.get("title") or node.get("text") or "")
            for node in nodes or []
        ).lower()
        return needle in haystack

    def compute_changed(
        self,
        prev_frame: str,
        next_frame: str,
        hash_distance: int,
        ssim_score: float | None,
        ax_changed: bool,
        visual_hash_static_threshold: int,
    ) -> bool:
        if ax_changed:
            return True
        if ssim_score is not None and ssim_score < self.settings.ssim_change_threshold:
            return True
        if hash_distance > visual_hash_static_threshold:
            return True
        return self.computer.has_changed(prev_frame, next_frame)

    @staticmethod
    def parse_expected_state(expected_state: str | None) -> tuple[str, str]:
        raw = str(expected_state or "").strip()
        if not raw:
            return "any", ""
        if ":" in raw:
            key, value = raw.split(":", 1)
            return key.strip().lower(), value.strip()
        return raw.strip().lower(), ""

    @staticmethod
    def has_non_clipboard_os_signal(before_snapshot: dict[str, Any], after_snapshot: dict[str, Any]) -> bool:
        keys = set(before_snapshot.keys()) | set(after_snapshot.keys())
        return any(str(key).strip().lower() != "clipboard" for key in keys)

    @staticmethod
    def has_non_clipboard_os_delta(before_snapshot: dict[str, Any], after_snapshot: dict[str, Any]) -> bool:
        keys = set(before_snapshot.keys()) | set(after_snapshot.keys())
        for key in keys:
            if str(key).strip().lower() == "clipboard":
                continue
            if before_snapshot.get(key) != after_snapshot.get(key):
                return True
        return False

    @staticmethod
    def is_os_telemetry_inconclusive_reason(reason: str) -> bool:
        return str(reason or "").strip().lower().startswith("os telemetry inconclusive")

    @staticmethod
    def is_a11y_unavailable_reason(reason: str) -> bool:
        token = str(reason or "").strip().lower()
        if not token:
            return True
        unavailable_markers = (
            "unavailable",
            "capture failed",
            "permission denied",
            "missing accessibility permission",
            "missing accessibility permissions",
            "accessibility permission",
            "accessibility permissions",
            "ax permission",
            "ax permissions",
            "ax api disabled",
            "kaxerrorapidisabled",
            "not trusted for accessibility",
            "not authorized for accessibility",
            "not supported",
            "blocked",
            "denied",
        )
        return any(marker in token for marker in unavailable_markers)

    @staticmethod
    def ax_changed(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
        if before is None or after is None:
            return False
        try:
            return json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)
        except Exception:
            return False

    def process_exists(self, name: str) -> bool:
        query = str(name or "").strip().lower()
        if not query:
            return False
        try:
            if self.computer.platform_name.lower().startswith("windows"):
                completed = subprocess.run(
                    ["tasklist", "/fo", "csv", "/nh"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            else:
                completed = subprocess.run(
                    ["ps", "-A", "-o", "comm="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
        except Exception:
            return False
        return query in (completed.stdout or "").lower()

    def _read_clipboard_snapshot(self) -> str | None:
        try:
            res = self.computer.execute({"type": "clipboard_op", "sub_action": "read"})
        except Exception:
            return None
        if not getattr(res, "success", False):
            return None
        metadata = getattr(res, "metadata", {}) or {}
        value = metadata.get("content")
        return "" if value is None else str(value)

    @staticmethod
    def _decode(image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _crop_region(image: Image.Image, frame: dict[str, Any], padding: int) -> Image.Image:
        x = int(float(frame.get("x", 0)) - padding)
        y = int(float(frame.get("y", 0)) - padding)
        w = int(float(frame.get("w", 0)) + padding * 2)
        h = int(float(frame.get("h", 0)) + padding * 2)
        left = max(0, x)
        top = max(0, y)
        right = min(image.width, x + max(1, w))
        bottom = min(image.height, y + max(1, h))
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _region_changed(before: Image.Image, after: Image.Image) -> bool:
        if before.size != after.size:
            after = after.resize(before.size)
        before_gray = before.convert("L")
        after_gray = after.convert("L")
        if skimage_ssim is not None:
            try:
                import numpy as np  # type: ignore

                score = skimage_ssim(np.array(before_gray), np.array(after_gray))
                return bool(score < 0.985)
            except Exception:
                pass
        before_pixels = list(before_gray.getdata())
        after_pixels = list(after_gray.getdata())
        if not before_pixels or len(before_pixels) != len(after_pixels):
            return False
        diff = sum(abs(a - b) for a, b in zip(before_pixels, after_pixels))
        max_diff = 255 * len(before_pixels)
        return (diff / max_diff) > 0.01
