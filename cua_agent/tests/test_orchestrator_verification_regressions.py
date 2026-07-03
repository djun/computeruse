"""Regression tests for post-action verification.

The verification logic lives in ``VerificationManager`` (the live path used by
the orchestrator loop). These tests target it directly. Element-reference
resolution still lives on ``Orchestrator`` and is tested against it.
"""

from cua_agent.agent.state_manager import StateManager, VerificationContract
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.orchestrator.verification_manager import VerificationManager
from cua_agent.utils.config import Settings


def _verifier() -> VerificationManager:
    # The methods under test are pure for these scenarios, so we can bypass __init__.
    return VerificationManager.__new__(VerificationManager)


def _orchestrator() -> Orchestrator:
    return Orchestrator.__new__(Orchestrator)


class _FakeComputer:
    """Minimal computer that reports a stable frame with a visual change.

    ``structural_similarity`` below the change threshold makes ``compute_changed``
    report a change without the settle loop needing many iterations.
    """

    platform_name = "Darwin"

    def capture_base64(self) -> str:
        return "frame-after"

    def capture_with_hash(self):
        return "frame-after", "hash-after"

    def hash_base64(self, frame: str) -> str:
        return "hash-after"

    def hash_distance(self, a, b) -> int:
        return 0

    def has_changed(self, a, b, threshold: float = 0.002) -> bool:
        # Stable across settle polls so the stabilization loop exits quickly.
        return False

    def structural_similarity(self, a, b) -> float:
        return 0.5  # < ssim_change_threshold -> counts as changed

    def get_active_window_tree(self, max_depth: int = 4):  # pragma: no cover - unused
        raise AssertionError("get_active_window_tree should not be called (enable_semantic=False)")

    def detect_ui_elements(self, frame):  # pragma: no cover - unused
        return []


def _verifier_with_fake_computer() -> VerificationManager:
    settings = Settings()
    settings.enable_semantic = False  # skip a11y capture; force pure-visual change detection
    return VerificationManager(settings, _FakeComputer())


def test_default_sensor_for_shell_action_is_none() -> None:
    verifier = _verifier()
    assert verifier.default_sensor_for_action({"type": "sandbox_shell"}) == "none"
    assert verifier.default_sensor_for_action({"type": "script_op"}) == "none"


def test_default_sensor_for_clipboard_depends_on_sub_action_and_open_app_stays_os_telemetry() -> None:
    verifier = _verifier()
    assert verifier.default_sensor_for_action({"type": "clipboard_op", "sub_action": "read"}) == "none"
    assert verifier.default_sensor_for_action({"type": "clipboard_op", "sub_action": "write"}) == "os_telemetry"
    assert verifier.default_sensor_for_action({"type": "clipboard_op", "sub_action": "clear"}) == "os_telemetry"
    assert verifier.default_sensor_for_action({"type": "open_app"}) == "os_telemetry"


def test_default_sensor_for_focus_window_and_wait_for_idle() -> None:
    verifier = _verifier()
    assert verifier.default_sensor_for_action({"type": "focus_window"}) == "os_telemetry"
    assert verifier.default_sensor_for_action({"type": "wait_for_idle"}) == "none"


def test_resolve_element_references_numeric_element_ref_uses_overlay_id() -> None:
    orchestrator = _orchestrator()
    action = {"type": "click_element", "element_ref": "7"}
    tags = [
        {
            "id": 7,
            "role": "AXButton",
            "label": "Submit",
            "path": "AXWindow > AXButton:Submit",
            "frame": {"x": 100.0, "y": 200.0, "w": 40.0, "h": 20.0},
        }
    ]

    resolved = orchestrator._resolve_element_references(action, tags)

    assert resolved is True
    assert action["x"] == 120.0
    assert action["y"] == 210.0
    assert action["semantic_role"] == "AXButton"
    assert action["semantic_label"] == "Submit"


def test_resolve_element_references_numeric_element_ref_missing_id_fails() -> None:
    orchestrator = _orchestrator()
    action = {"type": "click_element", "element_ref": "9"}
    tags = [{"id": 7, "frame": {"x": 1, "y": 2, "w": 3, "h": 4}}]

    resolved = orchestrator._resolve_element_references(action, tags)

    assert resolved is False


def test_clipboard_write_contract_defaults_to_clipboard_equals() -> None:
    verifier = _verifier()
    contract = verifier.resolve_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "write", "content": "hello"},
        current_step=None,
    )
    assert contract.sensor == "os_telemetry"
    assert contract.expected_state == "clipboard_equals:hello"


def test_clipboard_clear_contract_defaults_to_clipboard_equals_empty() -> None:
    verifier = _verifier()
    contract = verifier.resolve_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "clear"},
        current_step=None,
    )
    assert contract.sensor == "os_telemetry"
    assert contract.expected_state == "clipboard_equals:"


def test_clipboard_read_contract_defaults_to_no_verification() -> None:
    verifier = _verifier()
    contract = verifier.resolve_contract(
        state=StateManager(),
        action={"type": "clipboard_op", "sub_action": "read"},
        current_step=None,
    )
    assert contract.sensor == "none"
    assert contract.expected_state is None


def test_a11y_unavailable_matches_accessibility_permission_errors() -> None:
    verifier = _verifier()
    reason = "AX API disabled: process is not trusted for Accessibility permissions"
    assert verifier.is_a11y_unavailable_reason(reason) is True


def test_a11y_unavailable_does_not_match_regular_tree_mismatch() -> None:
    verifier = _verifier()
    assert verifier.is_a11y_unavailable_reason("a11y text not found") is False


def test_os_telemetry_any_is_inconclusive_without_non_clipboard_signal() -> None:
    verifier = _verifier()
    passed, reason = verifier.evaluate_os_telemetry_state(
        expected_state=None,
        before_snapshot={"clipboard": "unchanged"},
        after_snapshot={"clipboard": "unchanged"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_any_clipboard_only_delta_is_inconclusive() -> None:
    verifier = _verifier()
    passed, reason = verifier.evaluate_os_telemetry_state(
        expected_state="state_change",
        before_snapshot={"clipboard": "before"},
        after_snapshot={"clipboard": "after"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_any_passes_when_non_clipboard_delta_exists() -> None:
    verifier = _verifier()
    passed, reason = verifier.evaluate_os_telemetry_state(
        expected_state="state_change",
        before_snapshot={"clipboard": "before", "processes": ["calc.exe"]},
        after_snapshot={"clipboard": "after", "processes": ["notepad.exe"]},
    )
    assert passed is True
    assert reason == "os telemetry changed"


def test_os_telemetry_freeform_expected_is_inconclusive_without_non_clipboard_signal() -> None:
    verifier = _verifier()
    passed, reason = verifier.evaluate_os_telemetry_state(
        expected_state="calculator is focused",
        before_snapshot={"clipboard": "unchanged"},
        after_snapshot={"clipboard": "unchanged"},
    )
    assert passed is False
    assert "inconclusive" in reason


def test_verify_os_telemetry_short_circuits_when_inconclusive() -> None:
    verifier = _verifier()
    verifier.collect_os_telemetry_snapshot = lambda contract: {"clipboard": "same"}
    contract = VerificationContract(sensor="os_telemetry", expected_state=None, timeout_seconds=3)
    passed, reason = verifier.verify_os_telemetry(contract, {"clipboard": "same"})
    assert passed is False
    assert "inconclusive" in reason


def test_os_telemetry_inconclusive_uses_visual_fallback_result() -> None:
    verifier = _verifier()
    verifier.verify_os_telemetry = lambda contract, before: (
        False,
        "os telemetry inconclusive (no non-clipboard signal)",
    )
    verifier.run_visual_verification = lambda **kwargs: {
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
    outcome = verifier.run_verification_contract(
        action={"type": "open_app"},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        visual_hash_static_threshold=4,
    )
    assert outcome["passed"] is True
    assert outcome["sensor"] == "os_telemetry"
    assert outcome["note"] == "verification:os_telemetry:fallback"
    assert "visual fallback detected change" in str(outcome["reason"])


def test_vision_full_state_change_expected_uses_visual_change_without_a11y() -> None:
    # Real run_visual_verification with a fake computer: exercises the live
    # state_change path (no a11y evaluation) instead of a mocked stub.
    verifier = _verifier_with_fake_computer()
    contract = VerificationContract(sensor="vision_full", expected_state="state_change", timeout_seconds=3)
    outcome = verifier.run_verification_contract(
        action={"type": "click", "verify_after": True},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        visual_hash_static_threshold=4,
    )
    assert outcome["passed"] is True
    assert outcome["sensor"] == "vision_full"
    assert outcome["note"] == "verification:vision_full"
    assert outcome["reason"] == "vision_full detected change"


def test_vision_full_a11y_expected_uses_visual_fallback_when_tree_unavailable() -> None:
    # a11y-shaped expected_state on a vision_full contract: when the tree is
    # unavailable, the visual change carries the verification. The granular
    # ":fallback" note is produced by run_visual_verification.
    verifier = _verifier_with_fake_computer()
    contract = VerificationContract(sensor="vision_full", expected_state="text_exists:Settings", timeout_seconds=3)
    outcome = verifier.run_visual_verification(
        action={"type": "click", "verify_after": True},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        global_hotkeys=set(),
        visual_hash_static_threshold=4,
    )
    assert outcome["passed"] is True
    assert outcome["sensor"] == "vision_full"
    assert outcome["note"] == "verification:vision_full:fallback"
    assert "a11y tree unavailable; visual fallback detected change" in str(outcome["reason"])


def test_explicit_os_telemetry_failure_is_not_masked_by_fallback() -> None:
    verifier = _verifier()
    verifier.verify_os_telemetry = lambda contract, before: (False, "process not found")
    verifier.run_visual_verification = lambda **kwargs: {
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
    outcome = verifier.run_verification_contract(
        action={"type": "open_app"},
        contract=contract,
        current_frame="frame-before",
        current_hash="hash-before",
        ax_tree_before=None,
        telemetry_before={"clipboard": "same"},
        global_hotkeys=set(),
        visual_hash_static_threshold=4,
    )
    assert outcome["passed"] is False
    assert outcome["sensor"] == "os_telemetry"
    assert outcome["note"] == "verification:os_telemetry:timeout"
    assert outcome["reason"] == "process not found"
