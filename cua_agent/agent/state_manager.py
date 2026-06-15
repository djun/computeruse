"""Execution loop state tracking and action results."""

from __future__ import annotations

import base64
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cua_agent.orchestrator.react_types import GroundingBundle, ReActTurn

VERIFICATION_SENSOR_HIERARCHY: tuple[str, ...] = (
    "none",
    "os_telemetry",
    "a11y_tree",
    "pixel_diff",
    "vision_full",
)
DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 5
MAX_VERIFICATION_TIMEOUT_SECONDS = 30


@dataclass
class ActionResult:
    success: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationContract:
    sensor: str = "none"
    expected_state: str | None = None
    timeout_seconds: int = DEFAULT_VERIFICATION_TIMEOUT_SECONDS

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "sensor": self.sensor,
            "timeout_seconds": int(self.timeout_seconds),
        }
        if self.expected_state:
            payload["expected_state"] = self.expected_state
        return payload


@dataclass
class Observation:
    image_path: str
    timestamp: float
    changed_since_last: bool = False
    note: str = ""
    phash: str | None = None
    visual_hash: str | None = None
    hash_distance: int | None = None

    def __post_init__(self) -> None:
        if self.visual_hash is None and self.phash is not None:
            self.visual_hash = self.phash
        if self.phash is None and self.visual_hash is not None:
            self.phash = self.visual_hash


@dataclass
class Note:
    content: str
    source: str
    timestamp: float


class StateManager:
    """Tracks loop state, history, and termination criteria."""

    SENSOR_HIERARCHY = VERIFICATION_SENSOR_HIERARCHY

    def __init__(
        self,
        max_steps: int = 50,
        max_failures: int = 5,
        max_wall_clock_seconds: Optional[int] = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_failures = max_failures
        self.max_wall_clock_seconds = max_wall_clock_seconds

        self.history: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[Observation] = []
        self.notebook: List[Note] = []
        self.turns: List["ReActTurn"] = []
        self.event_log: List[Dict[str, Any]] = []
        self.last_grounding: "GroundingBundle | None" = None
        self.failure_count = 0
        self.steps = 0
        self.started_at = time.time()
        self.stuck_reasons: List[str] = []

    @classmethod
    def sensor_rank(cls, sensor: str) -> int:
        token = str(sensor or "").strip().lower()
        try:
            return cls.SENSOR_HIERARCHY.index(token)
        except ValueError:
            return len(cls.SENSOR_HIERARCHY)

    def normalize_verification_contract(
        self,
        raw_contract: Optional[Dict[str, Any]],
        *,
        fallback_sensor: str = "a11y_tree",
        fallback_expected_state: Optional[str] = None,
        verify_after: Optional[bool] = None,
    ) -> VerificationContract:
        candidate = raw_contract if isinstance(raw_contract, dict) else {}

        fallback = str(fallback_sensor or "a11y_tree").strip().lower()
        if fallback not in self.SENSOR_HIERARCHY:
            fallback = "a11y_tree"

        sensor = str(candidate.get("sensor") or fallback).strip().lower()
        if verify_after is False:
            sensor = "none"
        if sensor not in self.SENSOR_HIERARCHY:
            sensor = fallback

        expected_state_raw = candidate.get("expected_state")
        if expected_state_raw is None:
            expected_state_raw = fallback_expected_state
        expected_state = str(expected_state_raw).strip() if expected_state_raw is not None else None
        if expected_state == "":
            expected_state = None
        if expected_state and len(expected_state) > 500:
            expected_state = expected_state[:500]

        timeout_raw = candidate.get("timeout_seconds", DEFAULT_VERIFICATION_TIMEOUT_SECONDS)
        try:
            timeout_seconds = int(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = DEFAULT_VERIFICATION_TIMEOUT_SECONDS
        timeout_seconds = max(1, min(timeout_seconds, MAX_VERIFICATION_TIMEOUT_SECONDS))
        if sensor == "none":
            timeout_seconds = 1

        return VerificationContract(
            sensor=sensor,
            expected_state=expected_state,
            timeout_seconds=timeout_seconds,
        )

    def record_observation(
        self,
        image_b64: str,
        changed: bool,
        note: str = "",
        phash: str | None = None,
        visual_hash: str | None = None,
        hash_distance: int | None = None,
    ) -> Observation:
        # Offload image to disk to save memory
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"obs_{self.steps}_") as tmp:
                tmp.write(base64.b64decode(image_b64))
                image_path = tmp.name
        except Exception:
            # Fallback if disk write fails
            image_path = ""

        obs = Observation(
            image_path=image_path,
            timestamp=time.time(),
            changed_since_last=changed,
            note=note,
            phash=phash or visual_hash,
            visual_hash=visual_hash or phash,
            hash_distance=hash_distance,
        )
        self.observations.append(obs)
        self.history.append(
            f"observation@{obs.timestamp}:changed={changed}" + (f":{note}" if note else "")
        )
        return obs

    def add_note(self, content: str, source: str = "agent") -> None:
        """Add a persistent note to the working memory."""
        self.notebook.append(Note(content=content, source=source, timestamp=time.time()))
        self.history.append(f"notebook: added note from {source}")

    def get_notebook_summary(self) -> str:
        """Return a formatted string of all notes."""
        if not self.notebook:
            return "Notebook is empty."
        lines = ["Current Notebook Content:"]
        for i, note in enumerate(self.notebook, 1):
            lines.append(f"{i}. [{note.source}] {note.content}")
        return "\n".join(lines)

    def clear_notebook(self) -> None:
        self.notebook.clear()
        self.history.append("notebook: cleared")

    def record_action(self, action: Dict[str, Any], result: ActionResult) -> None:
        self.actions.append(action)
        action_summary = {
            "type": action.get("type", "unknown"),
            "success": result.success,
            "reason": result.reason,
            "keys": action.get("keys"),
            "text": action.get("text"),
            "x": action.get("x"),
            "y": action.get("y"),
            "cmd": action.get("cmd"),
            "operation": action.get("operation"),
            "path": action.get("path"),
            "execution": action.get("execution"),
        }
        self.history.append(f"action:{action_summary}")
        browser_summary = self._summarize_browser_result(action, result)
        if browser_summary:
            self.history.append(browser_summary)
        self.steps += 1
        if not result.success and result.reason != "hotkey deduped":
            self.failure_count += 1

    def should_halt(self) -> bool:
        if self.max_steps and self.steps >= self.max_steps:
            return True
        if self.max_failures and self.failure_count >= self.max_failures:
            return True
        if self.max_wall_clock_seconds and (time.time() - self.started_at) >= self.max_wall_clock_seconds:
            return True
        return False

    def record_stuck(self, reason: str) -> None:
        self.stuck_reasons.append(reason)
        self.history.append(f"stuck:{reason}")

    def record_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "type": str(event_type or "event"),
            "timestamp": time.time(),
            "payload": dict(payload or {}),
        }
        self.event_log.append(event)

    def record_grounding(self, grounding: "GroundingBundle") -> None:
        self.last_grounding = grounding
        self.record_event("grounding", grounding.to_compact_dict(include_image=False))

    def record_turn(self, turn: "ReActTurn") -> None:
        self.turns.append(turn)
        self.record_event("react_turn", turn.to_dict())

    def record_verification_failure(self, reason: str, action: Optional[Dict[str, Any]] = None) -> None:
        """Record a post-action verification failure as a real failure signal."""
        self.failure_count += 1
        action_type = (action or {}).get("type", "unknown")
        self.history.append(f"verification_failure:{action_type}:{reason}")

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "failures": self.failure_count,
            "history": list(self.history),
            "actions": list(self.actions),
            "observations": len(self.observations),
            "runtime_seconds": time.time() - self.started_at,
            "stuck_reasons": list(self.stuck_reasons),
        }

    def compact_view(self) -> Dict[str, Any]:
        last_action = self.actions[-1] if self.actions else None
        last_turn = self.turns[-1].to_dict() if self.turns else None
        grounding_quality = {}
        if self.last_grounding:
            grounding_quality = dict(self.last_grounding.quality)
        return {
            "steps": self.steps,
            "failure_count": self.failure_count,
            "stuck_reasons": list(self.stuck_reasons[-5:]),
            "last_action": last_action,
            "last_turn": last_turn,
            "grounding_quality": grounding_quality,
            "recent_events": list(self.event_log[-10:]),
            "recent_history": list(self.history[-12:]),
        }

    def to_react_view(self) -> Dict[str, Any]:
        return self.compact_view()

    def _summarize_browser_result(self, action: Dict[str, Any], result: ActionResult) -> str:
        """
        Push browser tool outputs into history so the LLM can read them on the next turn.
        Truncates large payloads to protect the prompt budget.
        """
        if action.get("execution") != "browser":
            return ""

        metadata = result.metadata or {}
        payload: Any = metadata.get("data")
        if payload is None:
            payload = metadata.get("raw") or metadata.get("output")

        # Unwrap common {"result": ...} structure from BrowserDriver
        if isinstance(payload, dict) and "result" in payload:
            payload = payload.get("result")

        if payload is None:
            return ""

        try:
            text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
        except Exception:
            text = str(payload)

        max_len = 1200
        if len(text) > max_len:
            text = text[:max_len] + "... [truncated]"

        cmd = action.get("command") or action.get("type") or "browser_result"
        return f"browser_result:{cmd}:{text}"
