"""Interaction-contract features: done challenged against the plan, target-aware
policy risk, the capability manifest, typed ActionResult codes, and ask_user.
"""

from types import SimpleNamespace

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.base_computer import DriverBackedComputerAdapter
from cua_agent.computer.types import COMPUTER_ACTION_SPACE, DisplayInfo
from cua_agent.orchestrator.action_policy import ActionPolicy
from cua_agent.orchestrator.planning import Plan, Step
from cua_agent.orchestrator.react_types import GroundingBundle
from cua_agent.orchestrator.recovery_manager import RecoveryManager
from cua_agent.agent.state_manager import ActionResult, StateManager
from cua_agent.utils.config import Settings

from cua_agent.tests.test_loop_control_actions import _orchestrator

_DISPLAY = DisplayInfo(
    logical_width=1280,
    logical_height=720,
    physical_width=1280,
    physical_height=720,
    scale_factor=1.0,
)


class _DummyComputer:
    platform_name = "test"
    system_info = "test"
    display = _DISPLAY


def _core() -> CognitiveCore:
    return CognitiveCore(Settings(use_openrouter=False), _DummyComputer())


def _plan() -> Plan:
    return Plan(
        id="p1",
        user_prompt="do the thing",
        steps=[Step(id=1, description="open the settings pane", success_criteria="settings visible")],
    )


# ---------------------------------------------------------------------------
# done challenged against the plan
# ---------------------------------------------------------------------------


def test_done_without_evidence_is_challenged_once_then_accepted(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "done", "reason": "finished"},
            {"type": "done", "reason": "finished, insisting"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task", plan=_plan())
    core = orch.cognitive_core
    assert len(core.visual_flags) == 2
    hint = (core.repeat_infos[1] or {}).get("hint", "")
    assert "evidence" in hint


def test_done_with_evidence_accepted_immediately_despite_plan(tmp_path) -> None:
    orch = _orchestrator(
        [{"type": "done", "reason": "finished", "evidence": "settings pane visible"}],
        tmp_path,
    )
    orch._run_session(user_prompt="test task", plan=_plan())
    assert len(orch.cognitive_core.visual_flags) == 1


def test_done_without_plan_accepted_immediately(tmp_path) -> None:
    orch = _orchestrator([{"type": "done", "reason": "finished"}], tmp_path)
    orch._run_session(user_prompt="test task")
    assert len(orch.cognitive_core.visual_flags) == 1


def test_done_challenge_is_per_step_not_once_per_session(tmp_path) -> None:
    # A premature done on step 1 is challenged; after advancing to step 2, a
    # second no-evidence done must be challenged again (not accepted because an
    # earlier, different step was already challenged).
    two_step_plan = Plan(
        id="p2",
        user_prompt="do it",
        steps=[
            Step(id=1, description="first goal", success_criteria="one done", status="in_progress"),
            Step(id=2, description="second goal", success_criteria="two done"),
        ],
    )
    orch = _orchestrator(
        [
            {"type": "done", "reason": "premature on step 1"},
            {"type": "left_click", "x": 1, "y": 2, "verification": {"sensor": "none", "timeout_seconds": 1}},
            {"type": "done", "reason": "premature on step 2"},
            {"type": "done", "reason": "insisting on step 2"},
        ],
        tmp_path,
    )
    orch.settings.strict_step_completion = False  # let the click advance step 1 -> 2
    orch._run_session(user_prompt="test task", plan=two_step_plan)
    # 4 proposals means the step-2 done was challenged (not short-circuited by the
    # step-1 challenge); the old session-global boolean would stop at 3.
    assert len(orch.cognitive_core.visual_flags) == 4


# ---------------------------------------------------------------------------
# Target-aware policy risk
# ---------------------------------------------------------------------------


def _grounding_with_tag(label: str, role: str = "AXButton") -> GroundingBundle:
    return GroundingBundle(
        screenshot_b64="ZnJhbWU=",
        frame_hash="hash",
        som_tags=[{"id": 5, "gid": "fused:5", "label": label, "role": role, "confidence": 0.9}],
    )


def _policy(hitl_enabled: bool = True) -> ActionPolicy:
    settings = Settings(use_openrouter=False)
    settings.autonomy_level = "confirm_risky"
    settings.enable_hitl_prompt = hitl_enabled
    return ActionPolicy(settings)


def test_destructive_label_escalates_to_hitl() -> None:
    decision = _policy().normalize_and_guard(
        {"type": "click_element", "element_id": 5},
        grounding=_grounding_with_tag("Delete account"),
        state=None,
    )
    assert decision.allowed is True
    assert decision.action.get("requires_hitl_confirmation") is True
    assert decision.risk_level == "high"
    assert "Delete account" in decision.action.get("hitl_reason", "")


def test_destructive_label_blocks_when_hitl_unavailable() -> None:
    decision = _policy(hitl_enabled=False).normalize_and_guard(
        {"type": "click_element", "element_id": 5},
        grounding=_grounding_with_tag("Empty Trash"),
        state=None,
    )
    assert decision.allowed is False


def test_clean_api_label_ref_is_checked_without_matching_tag() -> None:
    decision = _policy().normalize_and_guard(
        {"type": "click_element", "element_ref": "Uninstall"},
        grounding=GroundingBundle(screenshot_b64="ZnJhbWU=", frame_hash="hash", som_tags=[]),
        state=None,
    )
    assert decision.risk_level == "high"


def test_secure_field_text_entry_escalates() -> None:
    decision = _policy().normalize_and_guard(
        {"type": "fill_field", "element_id": 5, "text": "hunter2"},
        grounding=_grounding_with_tag("Password", role="AXSecureTextField"),
        state=None,
    )
    assert decision.risk_level == "high"
    assert decision.action.get("requires_hitl_confirmation") is True


def test_benign_label_stays_low_risk() -> None:
    decision = _policy().normalize_and_guard(
        {"type": "click_element", "element_id": 5},
        grounding=_grounding_with_tag("Search"),
        state=None,
    )
    assert decision.risk_level == "low"
    assert "requires_hitl_confirmation" not in decision.action


def test_destructive_sub_action_escalates_macro() -> None:
    decision = _policy().normalize_and_guard(
        {"type": "macro_actions", "actions": [{"type": "left_click", "element_id": 5}]},
        grounding=_grounding_with_tag("Format Disk"),
        state=None,
    )
    assert decision.risk_level == "high"


# ---------------------------------------------------------------------------
# Capability manifest
# ---------------------------------------------------------------------------


def _adapter(settings: Settings) -> DriverBackedComputerAdapter:
    adapter = DriverBackedComputerAdapter.__new__(DriverBackedComputerAdapter)
    adapter.settings = settings
    adapter.platform_name = "test"
    return adapter


def test_capabilities_default_local_gui() -> None:
    settings = Settings(use_openrouter=False)
    settings.simulation_mode = False
    settings.enable_hid = True
    caps = {cap.name: cap for cap in _adapter(settings).describe_capabilities()}
    assert caps["screenshot"].mode == "real"
    assert caps["gui_input"].mode == "real"


def test_capabilities_simulation_mode_is_dry_run() -> None:
    settings = Settings(use_openrouter=False)
    settings.simulation_mode = True
    caps = {cap.name: cap for cap in _adapter(settings).describe_capabilities()}
    assert caps["gui_input"].mode == "dry_run"
    assert caps["gui_input"].available is True


def test_capabilities_remote_cli_blocks_gui_and_allows_shell() -> None:
    settings = Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=True)
    caps = {cap.name: cap for cap in _adapter(settings).describe_capabilities()}
    assert caps["gui_input"].mode == "blocked"
    assert caps["browser_dom"].mode == "blocked"
    assert caps["shell"].mode == "real"


def test_cognitive_core_renders_capabilities_block() -> None:
    core = _core()
    core.computer = _adapter(Settings(use_openrouter=False))
    context = core._capabilities_context()
    assert "Current capabilities" in context
    assert "- screenshot: real" in context


def test_cognitive_core_capabilities_block_empty_without_adapter_support() -> None:
    assert _core()._capabilities_context() == ""


# ---------------------------------------------------------------------------
# Typed ActionResult
# ---------------------------------------------------------------------------


def test_record_action_includes_code_fields_in_history() -> None:
    state = StateManager()
    state.record_action(
        {"type": "click_element"},
        ActionResult(
            success=False,
            reason="element_id not found",
            code="target_not_found",
            category="grounding",
            retryable=True,
            suggested_next=["observe:fused"],
        ),
    )
    entry = state.history[-1]
    assert "target_not_found" in entry
    assert "observe:fused" in entry


def test_record_action_omits_code_fields_when_absent() -> None:
    state = StateManager()
    state.record_action({"type": "left_click"}, ActionResult(success=True, reason="ok"))
    assert "code" not in state.history[-1]


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------


def test_ask_user_in_action_space_and_mapping() -> None:
    assert "ask_user" in COMPUTER_ACTION_SPACE
    mapped = _core()._map_single_computer_action(
        {"action": "ask_user", "question": "Which account should I use?", "kind": "ambiguity"}
    )
    assert mapped["type"] == "ask_user"
    assert mapped["kind"] == "ambiguity"
    assert _core()._map_single_computer_action({"action": "ask_user"})["type"] == "invalid_action"


def test_ask_user_answer_feeds_back_and_forces_vision(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "wait", "seconds": 0, "verification": {"sensor": "none", "timeout_seconds": 1}},
            {"type": "ask_user", "question": "Proceed with account A?", "kind": "ambiguity"},
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._request_user_input = lambda question, kind="other": "yes, account A"
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    # wait leaves vision off for turn 2; the answered ask_user forces it for turn 3.
    assert core.visual_flags == [True, False, True]
    hint = (core.repeat_infos[2] or {}).get("hint", "")
    assert "yes, account A" in hint


def test_ask_user_without_user_counts_as_stall_and_continues(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "ask_user", "question": "Login for me?", "kind": "credential_required"},
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._request_user_input = lambda question, kind="other": None
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    assert len(core.visual_flags) == 2
    hint = (core.repeat_infos[1] or {}).get("hint", "")
    assert "No interactive user" in hint


def test_recovery_budget_without_step_requests_user_input() -> None:
    manager = RecoveryManager(Settings(use_openrouter=False))
    state = StateManager()
    state.failure_count = 99
    decision = manager.decide(
        state=state,
        plan=None,
        verification={"passed": True},
    )
    assert decision.request_user_input is True
    assert decision.replan is False
