"""Plan/step models shared by planners and orchestrators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    id: int
    description: str
    success_criteria: str
    status: str = "pending"  # pending|in_progress|done|failed
    notes: str = ""
    expected_state: str = ""
    recovery_steps: List[str] = field(default_factory=list)
    sub_steps: List[str] = field(default_factory=list)
    preferred_sensor: str = "a11y_tree"
    risk_level: str = "low"
    grounding_strategy: str = "semantic_first"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    id: str
    user_prompt: str
    steps: List[Step] = field(default_factory=list)
    current_step_index: int = 0

    def current_step(self) -> Optional[Step]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> None:
        if not self.steps:
            return
        if 0 <= self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].status = "done"
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.steps[self.current_step_index].status = "in_progress"
        else:
            self.current_step_index = len(self.steps)

    def fail_current(self, note: str) -> None:
        step = self.current_step()
        if step:
            step.status = "failed"
            step.notes = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_prompt": self.user_prompt,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Plan":
        steps = []
        allowed_step_keys = set(Step.__dataclass_fields__.keys())
        for raw_step in payload.get("steps", []):
            if not isinstance(raw_step, dict):
                continue
            step_payload = {key: value for key, value in raw_step.items() if key in allowed_step_keys}
            steps.append(Step(**step_payload))
        return cls(
            id=payload.get("id", ""),
            user_prompt=payload.get("user_prompt", ""),
            steps=steps,
            current_step_index=int(payload.get("current_step_index", 0)),
        )
