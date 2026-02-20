from cua_agent.agent.state_manager import StateManager, VerificationContract
from cua_agent.orchestrator.orchestrator import Orchestrator


def _orchestrator() -> Orchestrator:
    # These helpers are pure for the scenarios under test, so we can bypass __init__.
    return Orchestrator.__new__(Orchestrator)


def test_default_sensor_for_shell_action_is_none() -> None:
    orchestrator = _orchestrator()
    assert orchestrator._default_sensor_for_action({"type": "sandbox_shell"}) == "none"
    assert orchestrator._default_sensor_for_action({"type": "script_op"}) == "none"


def test_default_sensor_for_clipboard_depends_on_sub_action_and_open_app_stays_os_telemetry() -> None:
    orchestrator = _orchestrator()
    assert orchestrator._default_sensor_for_action({"type": "clipboard_op", "sub_action": "read"}) == "none"
    assert orchestrator._default_sensor_for_action({"type": "clipboard_op", "sub_action": "write"}) == "os_telemetry"
    assert orchestrator._default_sensor_for_action({"type": "clipboard_op", "sub_action": "clear"}) == "os_telemetry"
    assert orchestrator._default_sensor_for_action({"type": "open_app"}) == "os_telemetry"


def test_clipboard_write_contract_defaults_to_clipboard_equals() -> None:
    orchestrator = _orchestrator()
    contract = orchestrator._resolve_verification_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "write", "content": "hello"},
        current_step=None,
    )
    assert contract.sensor == "os_telemetry"
    assert contract.expected_state == "clipboard_equals:hello"


def test_clipboard_clear_contract_defaults_to_clipboard_equals_empty() -> None:
    orchestrator = _orchestrator()
    contract = orchestrator._resolve_verification_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "clear"},
        current_step=None,
    )
    assert contract.sensor == "os_telemetry"
    assert contract.expected_state == "clipboard_equals:"


def test_clipboard_read_contract_defaults_to_no_verification() -> None:
    orchestrator = _orchestrator()
    contract = orchestrator._resolve_verification_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "read"},
        current_step=None,
    )
    assert contract.sensor == "none"
    assert contract.expected_state is None


def test_a11y_unavailable_matches_accessibility_permission_errors() -> None:
    orchestrator = _orchestrator()
    reason = "AX API disabled: process is not trusted for Accessibility permissions"
    assert orchestrator._is_a11y_unavailable_reason(reason) is True


def test_a11y_unavailable_does_not_match_regular_tree_mismatch() -> None:
    orchestrator = _orchestrator()
    assert orchestrator._is_a11y_unavailable_reason("a11y text not found") is False


def test_os_telemetry_any_is_inconclusive_without_non_clipboard_signal() -> None:
    orchestrator = _orchestrator()
    passed, reason = orchestrator._evaluate_os_telemetry_state(
        expected_state=None,
        before_snapshot={"clipboard": "unchanged"},
        after_snapshot={"clipboard": "unchanged"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_any_clipboard_only_delta_is_inconclusive() -> None:
    orchestrator = _orchestrator()
    passed, reason = orchestrator._evaluate_os_telemetry_state(
        expected_state="state_change",
        before_snapshot={"clipboard": "before"},
        after_snapshot={"clipboard": "after"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_any_passes_when_non_clipboard_delta_exists() -> None:
    orchestrator = _orchestrator()
    passed, reason = orchestrator._evaluate_os_telemetry_state(
        expected_state="state_change",
        before_snapshot={"clipboard": "before", "processes": ["calc.exe"]},
        after_snapshot={"clipboard": "after", "processes": ["notepad.exe"]},
    )
    assert passed is True
    assert reason == "os telemetry changed"


def test_os_telemetry_freeform_expected_is_inconclusive_without_non_clipboard_signal() -> None:
    orchestrator = _orchestrator()
    passed, reason = orchestrator._evaluate_os_telemetry_state(
        expected_state="calculator is focused",
        before_snapshot={"clipboard": "unchanged"},
        after_snapshot={"clipboard": "unchanged"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_verify_os_telemetry_short_circuits_when_inconclusive() -> None:
    orchestrator = _orchestrator()
    orchestrator._collect_os_telemetry_snapshot = lambda contract: {"clipboard": "same"}
    contract = VerificationContract(sensor="os_telemetry", expected_state=None, timeout_seconds=3)
    passed, reason = orchestrator._verify_os_telemetry(contract, {"clipboard": "same"})
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_inconclusive_uses_visual_fallback_result() -> None:
    orchestrator = _orchestrator()
    orchestrator._verify_os_telemetry = lambda contract, before: (
        False,
        "os telemetry inconclusive (no non-clipboard signal)",
    )
    orchestrator._run_visual_verification = lambda **kwargs: {
        "passed": False,
        "reason": "visual changed",
        "sensor": "vision_full",
        "changed": True,
        "next_frame": "next-frame",
        "next_hash": "next-hash",
        "hash_distance": 5,
        "ssim_score": None,
        "ax_tree_after": None,
        "ax_changed": False,
        "note": "verification:visual",
        "force_vision_next_turn": True,
    }
    contract = VerificationContract(sensor="os_telemetry", expected_state=None, timeout_seconds=3)
    outcome = orchestrator._run_verification_contract(
        action={"type": "open_app"},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        phash_static_threshold=4,
    )
    assert outcome["passed"] is True
    assert outcome["sensor"] == "os_telemetry"
    assert outcome["note"] == "verification:os_telemetry:fallback"
    assert "visual fallback detected change" in str(outcome["reason"])


def test_vision_full_state_change_expected_uses_visual_change_without_a11y() -> None:
    orchestrator = _orchestrator()
    evaluate_calls = {"count": 0}

    def _unexpected_a11y_eval(*args, **kwargs):
        evaluate_calls["count"] += 1
        return False, "a11y tree unavailable"

    orchestrator._evaluate_a11y_state = _unexpected_a11y_eval
    orchestrator._run_visual_verification = lambda **kwargs: {
        "passed": False,
        "reason": "visual changed",
        "sensor": "vision_full",
        "changed": True,
        "next_frame": "next-frame",
        "next_hash": "next-hash",
        "hash_distance": 5,
        "ssim_score": None,
        "ax_tree_after": None,
        "ax_changed": False,
        "note": "verification:visual",
        "force_vision_next_turn": True,
    }
    contract = VerificationContract(sensor="vision_full", expected_state="state_change", timeout_seconds=3)
    outcome = orchestrator._run_verification_contract(
        action={"type": "click", "verify_after": True},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        phash_static_threshold=4,
    )
    assert evaluate_calls["count"] == 0
    assert outcome["passed"] is True
    assert outcome["sensor"] == "vision_full"
    assert outcome["note"] == "verification:vision_full"
    assert outcome["reason"] == "vision_full detected change"


def test_vision_full_a11y_expected_uses_visual_fallback_when_tree_unavailable() -> None:
    orchestrator = _orchestrator()
    orchestrator._run_visual_verification = lambda **kwargs: {
        "passed": False,
        "reason": "visual changed",
        "sensor": "vision_full",
        "changed": True,
        "next_frame": "next-frame",
        "next_hash": "next-hash",
        "hash_distance": 5,
        "ssim_score": None,
        "ax_tree_after": None,
        "ax_changed": False,
        "note": "verification:visual",
        "force_vision_next_turn": True,
    }
    contract = VerificationContract(sensor="vision_full", expected_state="text_exists:Settings", timeout_seconds=3)
    outcome = orchestrator._run_verification_contract(
        action={"type": "click", "verify_after": True},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        phash_static_threshold=4,
    )
    assert outcome["passed"] is True
    assert outcome["sensor"] == "vision_full"
    assert outcome["note"] == "verification:vision_full:fallback"
    assert "a11y tree unavailable; visual fallback detected change" in str(outcome["reason"])


def test_explicit_os_telemetry_failure_is_not_masked_by_fallback() -> None:
    orchestrator = _orchestrator()
    orchestrator._verify_os_telemetry = lambda contract, before: (False, "process not found")
    orchestrator._run_visual_verification = lambda **kwargs: {
        "passed": True,
        "reason": "visual changed",
        "sensor": "vision_full",
        "changed": True,
        "next_frame": "next-frame",
        "next_hash": "next-hash",
        "hash_distance": 5,
        "ssim_score": None,
        "ax_tree_after": None,
        "ax_changed": False,
        "note": "verification:visual",
        "force_vision_next_turn": True,
    }
    contract = VerificationContract(
        sensor="os_telemetry",
        expected_state="process_exists:calculator",
        timeout_seconds=3,
    )
    outcome = orchestrator._run_verification_contract(
        action={"type": "open_app"},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        phash_static_threshold=4,
    )
    assert outcome["passed"] is False
    assert outcome["sensor"] == "os_telemetry"
    assert outcome["note"] == "verification:os_telemetry:timeout"
    assert outcome["reason"] == "process not found"
