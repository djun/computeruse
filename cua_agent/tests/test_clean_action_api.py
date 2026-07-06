"""Clean action API: alias actions (click/input_text/press_keys/drag/focus/
clipboard) and the structured `target` field, mapped onto legacy action types.
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


def test_clean_aliases_in_action_space() -> None:
    for alias in ("click", "input_text", "press_keys", "drag", "focus", "clipboard"):
        assert alias in COMPUTER_ACTION_SPACE


# ---------------------------------------------------------------------------
# click
# ---------------------------------------------------------------------------


def test_click_with_target_element_id_maps_to_click_element() -> None:
    mapped = _core()._map_single_computer_action({"action": "click", "target": {"element_id": 12}})
    assert mapped["type"] == "click_element"
    assert mapped["element_ref"] == "12"


def test_click_with_target_label_maps_to_semantic_ref() -> None:
    mapped = _core()._map_single_computer_action({"action": "click", "target": {"label": "Search"}})
    assert mapped["type"] == "click_element"
    assert mapped["element_ref"] == "Search"


def test_click_with_target_coordinates() -> None:
    mapped = _core()._map_single_computer_action({"action": "click", "target": {"x": 100, "y": 200}})
    assert mapped["type"] == "click_element"
    assert mapped["x"] == 100.0
    assert mapped["y"] == 200.0


def test_click_supports_click_type() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "click", "target": {"element_id": 3}, "click_type": "right"}
    )
    assert mapped["type"] == "click_element"
    assert mapped["click_type"] == "right"


def test_click_without_target_is_invalid() -> None:
    mapped = _core()._map_single_computer_action({"action": "click"})
    assert mapped["type"] == "invalid_action"


def test_target_does_not_override_explicit_coordinates() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "click", "x": 5, "y": 6, "target": {"x": 100, "y": 200}}
    )
    assert mapped["x"] == 5.0
    assert mapped["y"] == 6.0


# ---------------------------------------------------------------------------
# input_text
# ---------------------------------------------------------------------------


def test_input_text_with_target_maps_to_click_and_type_without_submit() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "input_text", "target": {"label": "Search"}, "text": "abc"}
    )
    assert mapped["type"] == "click_and_type"
    assert mapped["element_ref"] == "Search"
    assert mapped["text"] == "abc"
    assert mapped["submit"] is False


def test_input_text_with_submit_opt_in() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "input_text", "target": {"label": "Search"}, "text": "abc", "submit": True}
    )
    assert mapped["type"] == "click_and_type"
    assert mapped["submit"] is True


def test_input_text_without_target_types_into_focused_field() -> None:
    mapped = _core()._map_single_computer_action({"action": "input_text", "text": "abc"})
    assert mapped["type"] == "type"
    assert mapped["text"] == "abc"


def test_input_text_without_target_with_submit_becomes_type_plus_enter() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "input_text", "text": "abc", "submit": True}
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["type", "key"]
    assert mapped["actions"][1]["keys"] == ["enter"]


def test_input_text_missing_text_is_invalid() -> None:
    mapped = _core()._map_single_computer_action({"action": "input_text"})
    assert mapped["type"] == "invalid_action"


def test_targetless_input_text_with_submit_flattens_inside_macro() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "click", "target": {"x": 10, "y": 20}},
                {"action": "input_text", "text": "abc", "submit": True},
            ]
        }
    )
    assert mapped["type"] == "macro_actions"
    assert [a["type"] for a in mapped["actions"]] == ["click_element", "type", "key"]


# ---------------------------------------------------------------------------
# other aliases
# ---------------------------------------------------------------------------


def test_press_keys_maps_to_key() -> None:
    mapped = _core()._map_single_computer_action({"action": "press_keys", "keys": ["ctrl", "s"]})
    assert mapped["type"] == "key"
    assert mapped["keys"] == ["ctrl", "s"]


def test_drag_maps_to_drag_and_drop() -> None:
    mapped = _core()._map_single_computer_action(
        {"action": "drag", "target": {"x": 10, "y": 20}, "target_x": 100, "target_y": 200}
    )
    assert mapped["type"] == "drag_and_drop"
    assert mapped["x"] == 10.0
    assert mapped["target_x"] == 100.0


def test_focus_maps_to_focus_window() -> None:
    mapped = _core()._map_single_computer_action({"action": "focus", "window_title": "Notes"})
    assert mapped["type"] == "focus_window"
    assert mapped["window_title"] == "Notes"


def test_clipboard_maps_to_clipboard_op() -> None:
    mapped = _core()._map_single_computer_action({"action": "clipboard", "sub_action": "read"})
    assert mapped["type"] == "clipboard_op"
    assert mapped["sub_action"] == "read"
