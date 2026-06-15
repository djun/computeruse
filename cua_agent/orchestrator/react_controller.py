"""Small controller helpers for ReAct turn bookkeeping."""

from __future__ import annotations

import time
from typing import Any

from cua_agent.agent.state_manager import StateManager
from cua_agent.orchestrator.planning import Plan
from cua_agent.orchestrator.react_types import ReActTurn
from cua_agent.utils.config import Settings


class ReactController:
    """Owns per-turn counters and visual refresh policy."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.turn_id = 0

    def start_turn(self, state: StateManager, plan: Plan | None) -> ReActTurn:
        current_step = plan.current_step() if plan else None
        turn = ReActTurn(
            turn_id=self.turn_id,
            step_id=current_step.id if current_step else None,
            observation_summary="",
            grounding_quality={},
            selected_target_gid=None,
            action=None,
            verification=None,
            result=None,
            reflection=None,
            recovery_decision=None,
            created_at=time.time(),
        )
        self.turn_id += 1
        return turn

    def should_force_vision(self, state: StateManager) -> bool:
        interval = max(0, int(getattr(self.settings, "force_visual_every_n_turns", 0) or 0))
        if interval and self.turn_id > 0 and self.turn_id % interval == 0:
            return True
        if state.last_grounding and state.last_grounding.quality.get("stale_hash"):
            return True
        return False

    def finalize_turn(
        self,
        turn: ReActTurn,
        *,
        observation_summary: str = "",
        grounding_quality: dict[str, Any] | None = None,
        selected_target_gid: str | None = None,
        action: dict[str, Any] | None = None,
        verification: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        reflection: dict[str, Any] | None = None,
        recovery_decision: dict[str, Any] | None = None,
    ) -> ReActTurn:
        turn.observation_summary = observation_summary
        turn.grounding_quality = dict(grounding_quality or {})
        turn.selected_target_gid = selected_target_gid
        turn.action = action
        turn.verification = verification
        turn.result = result
        turn.reflection = reflection
        turn.recovery_decision = recovery_decision
        return turn
