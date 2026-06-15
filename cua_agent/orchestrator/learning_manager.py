"""Learning hooks for ReAct turns and procedural memory."""

from __future__ import annotations

from cua_agent.agent.state_manager import StateManager
from cua_agent.orchestrator.react_types import ReActTurn


class LearningManager:
    """Currently records typed turns; skill synthesis remains in Orchestrator."""

    def record_turn(self, state: StateManager, turn: ReActTurn) -> None:
        state.record_turn(turn)
