"""Macros may not mix observation (or done) with executable sub-actions: the
macro is truncated at the first observation, which ends the turn and forces
fresh visual context before the model acts again.
"""

from typing import Any, Dict, List

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import DisplayInfo
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


# ---------------------------------------------------------------------------
# Mapping-layer behavior
# ---------------------------------------------------------------------------


def test_macro_truncated_at_observation_keeps_prefix_and_flags_observe_after() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "left_click", "x": 10, "y": 20},
                {"action": "screenshot"},
                {"action": "left_click", "x": 500, "y": 300},
            ]
        }
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["left_click"]
    assert mapped["observe_after"] is True
    assert "truncated" in mapped["truncation_note"]


def test_macro_starting_with_observation_returns_it_as_single_action() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "screenshot"},
                {"action": "left_click", "x": 500, "y": 300},
            ]
        }
    )
    assert mapped["type"] == "capture_only"


def test_macro_truncated_at_done_has_no_observe_after() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "left_click", "x": 10, "y": 20},
                {"action": "done", "reason": "finished"},
            ]
        }
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["left_click"]
    assert "observe_after" not in mapped
    assert "done" in mapped["truncation_note"]


def test_macro_without_observation_is_unchanged() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "left_click", "x": 10, "y": 20},
                {"action": "type", "text": "hello"},
            ]
        }
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["left_click", "type"]
    assert "observe_after" not in mapped
    assert "truncation_note" not in mapped


def test_macro_zoom_and_inspect_ui_also_truncate() -> None:
    core = _core()
    for observation in (
        {"action": "zoom", "region": [0, 0, 100, 100]},
        {"action": "inspect_ui"},
        {"action": "probe_ui", "x": 5, "y": 5},
    ):
        mapped = core._map_tool_args(
            {"actions": [{"action": "left_click", "x": 1, "y": 2}, observation]}
        )
        assert mapped["type"] == "macro_actions"
        assert mapped["observe_after"] is True


# ---------------------------------------------------------------------------
# Orchestrator behavior
# ---------------------------------------------------------------------------


def _macro(actions: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "macro_actions",
        "actions": actions,
        "verification": {"sensor": "none", "timeout_seconds": 1},
    }
    payload.update(extra)
    return payload


def test_macro_with_observe_after_forces_visual_context_next_turn(tmp_path) -> None:
    orch = _orchestrator(
        [
            _macro(
                [{"type": "left_click", "x": 1, "y": 2}],
                observe_after=True,
                truncation_note="macro truncated at 'capture_only'",
            ),
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    assert orch.cognitive_core.visual_flags == [True, True]


def test_macro_without_observe_after_keeps_sensor_none_semantics(tmp_path) -> None:
    orch = _orchestrator(
        [
            _macro([{"type": "left_click", "x": 1, "y": 2}]),
            {"type": "done", "reason": "finished"},
        ],
        tmp_path,
    )
    orch._run_session(user_prompt="test task")
    assert orch.cognitive_core.visual_flags == [True, False]
