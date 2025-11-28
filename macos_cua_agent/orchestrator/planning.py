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
        steps = [Step(**step) for step in payload.get("steps", [])]
        return cls(
            id=payload.get("id", ""),
            user_prompt=payload.get("user_prompt", ""),
            steps=steps,
            current_step_index=int(payload.get("current_step_index", 0)),
        )
