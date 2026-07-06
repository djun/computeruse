"""Loop-control semantics: `done` ends the task, `noop`/`invalid_action` keep the
loop alive with feedback, and `capture_only` ends the turn forcing fresh visual
context. Regression coverage for malformed actions silently terminating a run.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.types import COMPUTER_ACTION_SPACE, DisplayInfo
from cua_agent.memory.skill_composer import SkillComposer
from cua_agent.orchestrator.action_policy import ActionPolicy
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.orchestrator.react_controller import ReactController
from cua_agent.orchestrator.react_types import ActionEnvelope, GroundingBundle
from cua_agent.orchestrator.recovery_manager import RecoveryManager
from cua_agent.orchestrator.verification_manager import VerificationManager
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger

_DISPLAY = DisplayInfo(
    logical_width=1280,
    logical_height=720,
    physical_width=1280,
    physical_height=720,
    scale_factor=1.0,
)

_FRAME_B64 = "ZnJhbWU="  # "frame"


class _DummyComputer:
    platform_name = "test"
    system_info = "test"
    display = _DISPLAY


# ---------------------------------------------------------------------------
# Mapping-layer tests (CognitiveCore)
# ---------------------------------------------------------------------------


def _core() -> CognitiveCore:
    return CognitiveCore(Settings(use_openrouter=False), _DummyComputer())


def test_done_in_action_space() -> None:
    assert "done" in COMPUTER_ACTION_SPACE


def test_map_done_action_with_reason_and_evidence() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "done", "reason": "task finished", "evidence": "dialog closed"}
    )
    assert mapped["type"] == "done"
    assert mapped["reason"] == "task finished"
    assert mapped["evidence"] == "dialog closed"


def test_map_unknown_action_is_invalid_not_noop() -> None:
    mapped = _core()._map_single_computer_action({"action": "teleport"})
    assert mapped["type"] == "invalid_action"
    assert "teleport" in mapped["reason"]


def test_map_malformed_fill_field_is_invalid_not_noop() -> None:
    mapped = _core()._map_single_computer_action({"action": "fill_field"})
    assert mapped["type"] == "invalid_action"


def test_map_macro_with_only_invalid_subs_is_invalid_with_reasons() -> None:
    core = _core()
    mapped = core._map_tool_args({"actions": [{"action": "fill_field"}, {"action": "teleport"}]})
    assert mapped["type"] == "invalid_action"
    assert "fill_field missing text" in mapped["reason"]


def test_parse_text_only_response_maps_to_done_with_evidence() -> None:
    core = _core()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="All finished."))]
    )
    mapped = core._parse_tool_call(response)
    assert mapped["type"] == "done"
    assert mapped["evidence"] == "All finished."


def test_parse_bad_tool_args_maps_to_invalid_action() -> None:
    core = _core()
    call = SimpleNamespace(function=SimpleNamespace(name="computer", arguments="{not json"))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))]
    )
    mapped = core._parse_tool_call(response)
    assert mapped["type"] == "invalid_action"


# ---------------------------------------------------------------------------
# Orchestrator loop harness
# ---------------------------------------------------------------------------


class _FakeComputer:
    platform_name = "test"
    system_info = "test"
    display = _DISPLAY

    def __init__(self) -> None:
        self.executed: List[Dict[str, Any]] = []

    def capture_with_hash(self):
        return _FRAME_B64, "hash"

    def hash_base64(self, frame: str) -> str:
        return "hash"

    def hash_distance(self, a, b) -> int:
        return 0

    def has_changed(self, a, b, threshold: float = 0.002) -> bool:
        return False

    def structural_similarity(self, a, b) -> float:
        return 1.0

    def execute(self, action: Dict[str, Any]) -> ActionResult:
        self.executed.append(dict(action))
        return ActionResult(success=True, reason=str(action.get("reason") or action.get("type")))

    def detect_ui_elements(self, frame):
        return []


class _FakeGrounder:
    def observe(self, *, previous=None, force_vision=False, include_semantic=True, include_visual=True):
        return GroundingBundle(
            screenshot_b64=_FRAME_B64,
            frame_hash="hash",
            som_tags=[],
            quality={},
            overlay_b64=_FRAME_B64,
            ax_tree=None,
        )


class _FakeDashboard:
    def __init__(self) -> None:
        self.events: List[str] = []

    def start_session(self, *args, **kwargs) -> None:
        pass

    def push_event(self, message: str) -> None:
        self.events.append(str(message))

    def push_thought(self, *args, **kwargs) -> None:
        pass

    def push_action(self, *args, **kwargs) -> None:
        pass

    def push_verification(self, *args, **kwargs) -> None:
        pass

    def finish_session(self, *args, **kwargs) -> None:
        pass


class _ScriptedCore:
    """Returns pre-scripted actions and records visual-context flags per turn."""

    tokens_used = 0

    def __init__(self, actions: List[Dict[str, Any]]) -> None:
        self._actions = list(actions)
        self.visual_flags: List[bool] = []
        self.repeat_infos: List[Any] = []

    def propose_react_action(self, overlay_frame, history, **kwargs) -> ActionEnvelope:
        self.visual_flags.append(bool(kwargs.get("include_visual_context")))
        self.repeat_infos.append(kwargs.get("repeat_info"))
        if not self._actions:
            return ActionEnvelope(action={"type": "done", "reason": "script exhausted"})
        return ActionEnvelope(action=dict(self._actions.pop(0)))


def _orchestrator(scripted: List[Dict[str, Any]], tmp_path) -> Orchestrator:
    settings = Settings(use_openrouter=False)
    settings.enable_semantic = False
    settings.max_total_tokens = None
    settings.enable_hitl_prompt = False
    settings.autonomy_level = "fully_autonomous"

    computer = _FakeComputer()
    orch = Orchestrator.__new__(Orchestrator)
    orch.settings = settings
    orch.logger = get_logger("test_loop_control", level="ERROR")
    orch.computer = computer
    orch.cognitive_core = _ScriptedCore(scripted)
    orch.memory = SimpleNamespace(
        search_skills=lambda query: [],
        search_skills_scored=lambda query: [],
        logs_dir=tmp_path,
        save_episode=lambda episode: None,
    )
    orch.planner = SimpleNamespace(tokens_used=0, summarize_episode=lambda *a, **k: "episode")
    orch.reflector = SimpleNamespace(available=False, tokens_used=0)
    orch.grounder = _FakeGrounder()
    orch.verifier = VerificationManager(settings, computer)
    orch.action_policy = ActionPolicy(settings)
    orch.react_controller = ReactController(settings)
    orch.recovery_manager = RecoveryManager(settings)
    orch.skill_composer = SkillComposer(platform_name="test")
    orch.trajectory = None
    orch.dashboard = _FakeDashboard()
    orch.display = computer.display
    orch.global_hotkeys = set()
    return orch


def test_done_stops_loop_and_records_reason(tmp_path) -> None:
    orch = _orchestrator([{"type": "done", "reason": "all steps complete", "evidence": "final screen"}], tmp_path)
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    assert len(core.visual_flags) == 1
    # done never reaches the computer
    assert orch.computer.executed == []


def test_invalid_action_feeds_back_and_loop_continues(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "invalid_action", "reason": "fill_field missing text"},
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    # The invalid action did NOT end the run: a second proposal happened.
    assert len(core.visual_flags) == 2
    # The second turn carried the error back to the model.
    hint = (core.repeat_infos[1] or {}).get("hint", "")
    assert "fill_field missing text" in hint


def test_noop_continues_instead_of_stopping(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "noop", "reason": "waiting for safe state"},
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    assert len(orch.cognitive_core.visual_flags) == 2


def test_consecutive_stalls_break_loop(tmp_path) -> None:
    orch = _orchestrator(
        [
            {"type": "invalid_action", "reason": "bad args 1"},
            {"type": "noop", "reason": "no safe action"},
            {"type": "invalid_action", "reason": "bad args 2"},
            {"type": "done", "reason": "should never be reached"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    # Loop stopped after the third consecutive stall, before consuming `done`.
    assert len(core.visual_flags) == 3
    assert any("stalled" in event for event in orch.dashboard.events)


def test_capture_only_forces_visual_context_next_turn(tmp_path) -> None:
    # wait uses the sensor-`none` verification path, which leaves
    # force_vision_next_turn=False -> turn 2 has no image. capture_only must
    # flip it back so turn 3 carries fresh visual context.
    orch = _orchestrator(
        [
            {"type": "wait", "seconds": 0, "verification": {"sensor": "none", "timeout_seconds": 1}},
            {"type": "capture_only", "reason": "model requested screenshot"},
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    core = orch.cognitive_core
    assert core.visual_flags == [True, False, True]
    # capture_only was executed (recorded) and acknowledged in history/events.
    assert any(action.get("type") == "capture_only" for action in orch.computer.executed)
