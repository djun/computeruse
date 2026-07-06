"""The unified `observe` action: one model-facing entry point for requesting
fresh context, mapped internally to capture_only / inspect_ui / zoom / noop.
"""

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import COMPUTER_ACTION_SPACE, DisplayInfo
from cua_agent.utils.config import Settings

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


def test_observe_in_action_space() -> None:
    assert "observe" in COMPUTER_ACTION_SPACE


def test_observe_defaults_to_auto_capture() -> None:
    mapped = _core()._map_single_computer_action({"action": "observe"})
    assert mapped["type"] == "capture_only"
    assert mapped["observe_mode"] == "auto"


def test_observe_screenshot_maps_to_capture_only_with_reason() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "observe", "mode": "screenshot", "reason": "need to see if dialog appeared"}
    )
    assert mapped["type"] == "capture_only"
    assert mapped["reason"] == "need to see if dialog appeared"
    assert mapped["observe_mode"] == "screenshot"


def test_observe_fused_maps_to_capture_only() -> None:
    mapped = _core()._map_single_computer_action({"action": "observe", "mode": "fused"})
    assert mapped["type"] == "capture_only"
    assert mapped["observe_mode"] == "fused"


def test_observe_ui_tree_maps_to_inspect_ui() -> None:
    mapped = _core()._map_single_computer_action({"action": "observe", "mode": "ui_tree"})
    assert mapped["type"] == "inspect_ui"


def test_observe_zoom_delegates_to_zoom_mapping() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "observe", "mode": "zoom", "region": [10, 20, 110, 90]}
    )
    assert mapped["type"] == "zoom"
    assert mapped["region"] == [10.0, 20.0, 110.0, 90.0]


def test_observe_zoom_without_region_is_invalid() -> None:
    mapped = _core()._map_single_computer_action({"action": "observe", "mode": "zoom"})
    assert mapped["type"] == "invalid_action"


def test_observe_none_maps_to_noop() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "observe", "mode": "none", "reason": "state already known"}
    )
    assert mapped["type"] == "noop"
    assert "state already known" in mapped["reason"]


def test_observe_unknown_mode_is_invalid() -> None:
    mapped = _core()._map_single_computer_action({"action": "observe", "mode": "xray"})
    assert mapped["type"] == "invalid_action"


def test_observe_inside_macro_truncates_with_observe_after() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "left_click", "x": 10, "y": 20},
                {"action": "observe", "mode": "screenshot"},
                {"action": "type", "text": "never runs"},
            ]
        }
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["left_click"]
    assert mapped["observe_after"] is True


def test_observe_ui_tree_default_sensor_is_none() -> None:
    assert _core()._default_sensor_for_action_type("observe") == "none"
