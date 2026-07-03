"""Pure repeat-stagnation logic extracted from the session loop."""

from cua_agent.orchestrator.orchestrator import Orchestrator

_apply = Orchestrator._apply_repeat_stagnation


def test_same_action_thrice_triggers_break() -> None:
    same, no_change, brk, reason = _apply(
        is_wait=False, pending_break=False, break_reason="",
        action_sig="a", last_action_sig="a", changed=True,
        repeat_same_action=2, repeat_without_change=0,
    )
    assert same == 3
    assert brk is True
    assert reason == "repeat_same_action:3"


def test_no_change_repeat_triggers_break() -> None:
    same, no_change, brk, reason = _apply(
        is_wait=False, pending_break=False, break_reason="",
        action_sig="a", last_action_sig="a", changed=False,
        repeat_same_action=0, repeat_without_change=1,
    )
    assert no_change == 2
    assert brk is True
    assert reason == "repeat_without_change"


def test_new_action_resets_counters() -> None:
    same, no_change, brk, reason = _apply(
        is_wait=False, pending_break=False, break_reason="",
        action_sig="b", last_action_sig="a", changed=True,
        repeat_same_action=2, repeat_without_change=1,
    )
    assert same == 0
    assert no_change == 0
    assert brk is False


def test_wait_action_resets_and_does_not_break() -> None:
    same, no_change, brk, reason = _apply(
        is_wait=True, pending_break=False, break_reason="",
        action_sig="a", last_action_sig="a", changed=False,
        repeat_same_action=2, repeat_without_change=1,
    )
    assert (same, no_change, brk) == (0, 0, False)


def test_already_pending_break_preserves_reason_and_resets() -> None:
    same, no_change, brk, reason = _apply(
        is_wait=False, pending_break=True, break_reason="critical_no_change",
        action_sig="a", last_action_sig="a", changed=False,
        repeat_same_action=2, repeat_without_change=1,
    )
    assert (same, no_change) == (0, 0)
    assert brk is True
    assert reason == "critical_no_change"
