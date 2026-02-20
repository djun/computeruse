from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.config import Settings


class _DummyComputer:
    platform_name = "test"
    system_info = "test"
    display = DisplayInfo(
        logical_width=1280,
        logical_height=720,
        physical_width=1280,
        physical_height=720,
        scale_factor=1.0,
    )


def _tool_names(core: CognitiveCore) -> list[str]:
    return [tool["function"]["name"] for tool in core._available_tools()]


def test_shell_tool_hidden_when_enable_shell_false() -> None:
    core = CognitiveCore(
        Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=False),
        _DummyComputer(),
    )

    names = _tool_names(core)
    assert "shell" not in names
    assert "script" not in names
    assert "notebook" in names


def test_shell_tool_visible_when_enable_shell_true() -> None:
    core = CognitiveCore(
        Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=True),
        _DummyComputer(),
    )

    names = _tool_names(core)
    assert "shell" in names
    assert "script" in names
    assert "notebook" in names


def test_map_shell_args_returns_noop_when_shell_disabled() -> None:
    core = CognitiveCore(
        Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=False),
        _DummyComputer(),
    )

    result = core._map_shell_args({"command": "echo hello"})

    assert result["type"] == "noop"
    assert "ENABLE_SHELL=false" in result["reason"]


def test_map_script_args_builds_script_op_payload() -> None:
    core = CognitiveCore(
        Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=True),
        _DummyComputer(),
    )

    result = core._map_script_args(
        {
            "action": "run",
            "path": "tools/task.py",
            "args": ["--dry-run"],
            "runtime_seconds": 8,
        }
    )

    assert result["type"] == "script_op"
    assert result["operation"] == "run"
    assert result["path"] == "tools/task.py"
    assert result["args"] == ["--dry-run"]
    assert result["runtime_seconds"] == 8
    assert result["execution"] == "shell"


def test_map_script_args_returns_noop_when_shell_disabled() -> None:
    core = CognitiveCore(
        Settings(use_openrouter=False, execution_profile="remote_cli", enable_shell=False),
        _DummyComputer(),
    )

    result = core._map_script_args({"action": "read", "path": "tools/task.py"})

    assert result["type"] == "noop"
    assert "ENABLE_SHELL=false" in result["reason"]
