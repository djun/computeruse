"""Agent core (model prompts, state, and action mapping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from cua_agent.agent.cognitive_core import CognitiveCore
    from cua_agent.agent.state_manager import ActionResult, StateManager

__all__ = ["ActionResult", "CognitiveCore", "StateManager"]


def __getattr__(name: str):  # pragma: no cover
    """
    Lazy attribute access to avoid circular imports during package initialization.

    Importing `cua_agent.agent.state_manager` executes this module; importing heavy
    modules here would create cycles with `cua_agent.computer.adapter`.
    """
    if name == "CognitiveCore":
        from cua_agent.agent.cognitive_core import CognitiveCore as value

        return value
    if name == "StateManager":
        from cua_agent.agent.state_manager import StateManager as value

        return value
    if name == "ActionResult":
        from cua_agent.agent.state_manager import ActionResult as value

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
