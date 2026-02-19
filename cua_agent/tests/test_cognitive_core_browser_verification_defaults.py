from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.utils.config import Settings


def _core(*, platform_name: str, windows_cyborg_mode: bool) -> CognitiveCore:
    core = CognitiveCore.__new__(CognitiveCore)
    core.settings = Settings(use_openrouter=False, windows_cyborg_mode=windows_cyborg_mode)
    core.platform_name = platform_name
    return core


def test_read_only_browser_ops_default_to_none_verification() -> None:
    core = _core(platform_name="macOS", windows_cyborg_mode=False)
    for command in ("get_page_content", "get_links", "get_dom_tree"):
        payload = core._map_browser_args({"command": command})
        assert payload["type"] == "browser_op"
        assert payload["verification"]["sensor"] == "none"
        assert payload["verification"]["timeout_seconds"] == 1


def test_browser_mutating_ops_keep_pixel_diff_default() -> None:
    core = _core(platform_name="macOS", windows_cyborg_mode=False)
    payload = core._map_browser_args({"command": "click_element", "selector": "#submit"})
    assert payload["type"] == "browser_op"
    assert payload["verification"]["sensor"] == "pixel_diff"
    assert payload["verification"]["timeout_seconds"] == 6


def test_read_only_browser_op_honors_explicit_verification_contract() -> None:
    core = _core(platform_name="macOS", windows_cyborg_mode=False)
    payload = core._map_browser_args(
        {
            "command": "get_links",
            "verification": {
                "sensor": "a11y_tree",
                "expected_state": "text_exists:Settings",
                "timeout_seconds": 7,
            },
        }
    )
    assert payload["verification"]["sensor"] == "a11y_tree"
    assert payload["verification"]["expected_state"] == "text_exists:Settings"
    assert payload["verification"]["timeout_seconds"] == 7


def test_windows_cyborg_navigate_uses_hostname_url_expectation() -> None:
    core = _core(platform_name="Windows 11", windows_cyborg_mode=True)
    payload = core._map_browser_args(
        {
            "command": "navigate",
            "url": "http://WWW.Example.com/some/path?q=1",
        }
    )
    assert payload["type"] == "macro_actions"
    assert payload["verification"]["sensor"] == "a11y_tree"
    assert payload["verification"]["expected_state"] == "url_contains:example.com"
    assert payload["verification"]["timeout_seconds"] == 8

