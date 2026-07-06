"""expected_effect: the model declares WHAT should happen; the runtime picks the
sensor and expected_state used to verify it. Explicit verification contracts win.
"""

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import contract_from_expected_effect
from cua_agent.computer.types import DisplayInfo
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


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------


def test_text_appears_uses_a11y_text_exists() -> None:
    contract = contract_from_expected_effect({"kind": "text_appears", "value": "Dashboard"})
    assert contract == {"sensor": "a11y_tree", "expected_state": "text_exists:Dashboard"}


def test_text_disappears_requires_value() -> None:
    assert contract_from_expected_effect({"kind": "text_disappears"}) is None
    contract = contract_from_expected_effect({"kind": "text_disappears", "value": "Loading"})
    assert contract == {"sensor": "a11y_tree", "expected_state": "text_not_exists:Loading"}


def test_clipboard_changed_with_and_without_value() -> None:
    assert contract_from_expected_effect({"kind": "clipboard_changed"}) == {
        "sensor": "os_telemetry",
        "expected_state": "clipboard_changed",
    }
    assert contract_from_expected_effect({"kind": "clipboard_changed", "value": "abc"}) == {
        "sensor": "os_telemetry",
        "expected_state": "clipboard_contains:abc",
    }


def test_app_opened_uses_process_exists() -> None:
    contract = contract_from_expected_effect({"kind": "app_opened", "value": "Calculator"})
    assert contract == {"sensor": "os_telemetry", "expected_state": "process_exists:Calculator"}


def test_file_created_uses_file_exists() -> None:
    contract = contract_from_expected_effect({"kind": "file_created", "value": "/tmp/out.pdf"})
    assert contract == {"sensor": "os_telemetry", "expected_state": "file_exists:/tmp/out.pdf"}


def test_page_navigated_prefers_url_contains() -> None:
    contract = contract_from_expected_effect({"kind": "page_navigated", "value": "example.com"})
    assert contract == {"sensor": "a11y_tree", "expected_state": "url_contains:example.com"}
    fallback = contract_from_expected_effect({"kind": "page_navigated"})
    assert fallback == {"sensor": "pixel_diff", "expected_state": "state_change"}


def test_visual_changed_and_no_change() -> None:
    assert contract_from_expected_effect({"kind": "visual_changed"}) == {
        "sensor": "pixel_diff",
        "expected_state": "state_change",
    }
    assert contract_from_expected_effect({"kind": "no_change"}) == {"sensor": "none"}


def test_timeout_passthrough_and_unknown_kind() -> None:
    contract = contract_from_expected_effect(
        {"kind": "text_appears", "value": "OK", "timeout_seconds": 12}
    )
    assert contract["timeout_seconds"] == 12
    assert contract_from_expected_effect({"kind": "levitate"}) is None
    assert contract_from_expected_effect(None) is None


# ---------------------------------------------------------------------------
# Mapping integration
# ---------------------------------------------------------------------------


def test_action_with_expected_effect_gets_derived_contract() -> None:
    mapped = _core()._map_single_computer_action(
        {
            "action": "left_click",
            "x": 10,
            "y": 20,
            "expected_effect": {"kind": "text_appears", "value": "Dashboard"},
        }
    )
    assert mapped["verification"]["sensor"] == "a11y_tree"
    assert mapped["verification"]["expected_state"] == "text_exists:Dashboard"


def test_explicit_verification_wins_over_expected_effect() -> None:
    mapped = _core()._map_single_computer_action(
        {
            "action": "left_click",
            "x": 10,
            "y": 20,
            "expected_effect": {"kind": "text_appears", "value": "Dashboard"},
            "verification": {"sensor": "pixel_diff", "timeout_seconds": 3},
        }
    )
    assert mapped["verification"]["sensor"] == "pixel_diff"


def test_macro_with_expected_effect_gets_derived_contract() -> None:
    mapped = _core()._map_tool_args(
        {
            "actions": [
                {"action": "left_click", "x": 1, "y": 2},
                {"action": "type", "text": "hello"},
            ],
            "expected_effect": {"kind": "app_focused", "value": "Notes"},
        }
    )
    assert mapped["type"] == "macro_actions"
    assert mapped["verification"]["sensor"] == "os_telemetry"
    assert mapped["verification"]["expected_state"] == "app_focused:Notes"
