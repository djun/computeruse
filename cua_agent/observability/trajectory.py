"""Deterministic trajectory recording and replay.

A trajectory is a JSONL log of turns (action + outcome + frame hash). Recording
is opt-in; replay re-executes the recorded actions against a computer adapter so
runs can be reproduced for debugging and regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


class TrajectoryRecorder:
    """Appends one JSON line per turn to a trajectory file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def record(self, action: dict[str, Any], *, success: bool, frame_hash: str | None = None,
               reason: str = "", note: str = "") -> None:
        entry = {
            "turn": self._index,
            "action": self._strip(action),
            "success": bool(success),
            "frame_hash": frame_hash,
            "reason": reason,
            "note": note,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._index += 1

    @staticmethod
    def _strip(action: dict[str, Any]) -> dict[str, Any]:
        # Drop heavy/base64 payloads that bloat the log and aren't needed to replay.
        drop = {"zoom_image", "screenshot_b64", "overlay_b64", "ax_tree_after", "next_frame"}
        return {k: v for k, v in (action or {}).items() if k not in drop and not str(k).startswith("_debug")}


def load_trajectory(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def replay_trajectory(
    computer: Any,
    records: Iterable[dict[str, Any]],
    *,
    on_step: Optional[Callable[[int, dict[str, Any], Any], None]] = None,
) -> list[dict[str, Any]]:
    """Re-execute each recorded action against `computer`. Returns per-step outcomes."""
    outcomes: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        action = record.get("action") or {}
        if not action or action.get("type") in {"noop", "capture_only", "done", "invalid_action", "ask_user"}:
            outcomes.append({"turn": idx, "skipped": True, "action": action})
            continue
        result = computer.execute(action)
        replay_success = bool(getattr(result, "success", False))
        original_success = bool(record.get("success", False))
        outcome = {
            "turn": idx,
            "action": action,
            "replay_success": replay_success,
            "original_success": original_success,
            "matched": replay_success == original_success,
        }
        outcomes.append(outcome)
        if on_step is not None:
            on_step(idx, action, result)
    return outcomes
