"""Execution loop state tracking and action results."""

from __future__ import annotations

import base64
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ActionResult:
    success: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    image_path: str
    timestamp: float
    changed_since_last: bool = False
    note: str = ""
    phash: str | None = None
    hash_distance: int | None = None


@dataclass
class Note:
    content: str
    source: str
    timestamp: float


class StateManager:
    """Tracks loop state, history, and termination criteria."""

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
        self.failure_count = 0
        self.steps = 0
        self.started_at = time.time()
        self.stuck_reasons: List[str] = []

    def record_observation(
        self,
        image_b64: str,
        changed: bool,
        note: str = "",
        phash: str | None = None,
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
            phash=phash,
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
