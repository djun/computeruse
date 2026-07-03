"""RecoveryDecision contract — the loop now consumes replan / force_vision_next_turn."""

from types import SimpleNamespace

from cua_agent.orchestrator.recovery_manager import RecoveryManager
from cua_agent.utils.config import Settings


def _mgr() -> RecoveryManager:
    return RecoveryManager(Settings())


def _plan_with_step():
    step = SimpleNamespace(status="in_progress")
    return SimpleNamespace(current_step=lambda: step)


def test_failed_verification_requests_visual_refresh() -> None:
    decision = _mgr().decide(
        state=SimpleNamespace(failure_count=0),
        plan=_plan_with_step(),
        verification={"passed": False, "reason": "no change"},
        repeat_same_action=0,
        repeat_without_change=0,
        reason="no change",
    )
    assert decision.refresh_grounding is True
    assert decision.force_vision_next_turn is True


def test_repeated_same_action_requests_replan() -> None:
    decision = _mgr().decide(
        state=SimpleNamespace(failure_count=0),
        plan=_plan_with_step(),
        verification={"passed": True},
        repeat_same_action=2,  # >= max_same_target_failures (default 2)
        repeat_without_change=0,
        reason="",
    )
    assert decision.replan is True


def test_failure_budget_requests_replan() -> None:
    decision = _mgr().decide(
        state=SimpleNamespace(failure_count=3),  # >= max_recovery_attempts_per_step (default 3)
        plan=_plan_with_step(),
        verification={"passed": True},
        repeat_same_action=0,
        repeat_without_change=0,
        reason="",
    )
    assert decision.replan is True


def test_healthy_turn_is_continue() -> None:
    decision = _mgr().decide(
        state=SimpleNamespace(failure_count=0),
        plan=_plan_with_step(),
        verification={"passed": True},
        repeat_same_action=0,
        repeat_without_change=0,
        reason="ok",
    )
    assert decision.replan is False
    assert decision.refresh_grounding is False
    assert decision.force_vision_next_turn is False
