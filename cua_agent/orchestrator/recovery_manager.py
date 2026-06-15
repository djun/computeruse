"""Recovery decisions for failed or stagnant ReAct turns."""

from __future__ import annotations

from typing import Any

from cua_agent.agent.state_manager import StateManager
from cua_agent.orchestrator.planning import Plan
from cua_agent.orchestrator.react_types import RecoveryDecision
from cua_agent.utils.config import Settings


class RecoveryManager:
    """Converts verification and loop state into recovery decisions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(
        self,
        *,
        state: StateManager,
        plan: Plan | None,
        verification: dict[str, Any] | None,
        repeat_same_action: int = 0,
        repeat_without_change: int = 0,
        reason: str = "",
    ) -> RecoveryDecision:
        current_step = plan.current_step() if plan else None
        if verification and not bool(verification.get("passed", True)):
            return RecoveryDecision(
                refresh_grounding=True,
                force_vision_next_turn=True,
                reason=str(verification.get("reason") or reason or "verification failed"),
            )
        if repeat_same_action >= int(getattr(self.settings, "max_same_target_failures", 2)):
            return RecoveryDecision(replan=True, refresh_grounding=True, reason="same target repeated")
        if repeat_without_change >= int(getattr(self.settings, "max_same_target_failures", 2)):
            return RecoveryDecision(replan=bool(current_step), refresh_grounding=True, reason="no progress")
        if state.failure_count >= int(getattr(self.settings, "max_recovery_attempts_per_step", 3)):
            return RecoveryDecision(replan=bool(current_step), reason="recovery budget reached")
        return RecoveryDecision(reason=reason or "continue")
