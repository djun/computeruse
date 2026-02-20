import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from cua_agent.agent.cognitive_core import BROWSER_TOOL, CognitiveCore, ToolRegistration
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


def _tool_response(tool_name: str, args: dict) -> SimpleNamespace:
    function = SimpleNamespace(name=tool_name, arguments=json.dumps(args))
    tool_call = SimpleNamespace(function=function)
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_parse_tool_call_dispatches_via_registry_for_builtin_tool() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    response = _tool_response("notebook", {"action": "add_note", "content": "fact"})

    parsed = core._parse_tool_call(response)

    assert parsed["type"] == "notebook_op"
    assert parsed["action"] == "add_note"
    assert parsed["content"] == "fact"


def test_register_tool_allows_custom_tool_schema_and_mapper() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    custom_schema = {
        "type": "function",
        "function": {
            "name": "workspace",
            "description": "workspace helper",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
    core.register_tool(
        ToolRegistration(
            name="workspace",
            schema=custom_schema,
            enabled=lambda _core: True,
            mapper=lambda _core, args: {"type": "workspace_op", "path": args.get("path", "")},
        )
    )

    tool_names = [tool["function"]["name"] for tool in core._available_tools()]
    parsed = core._parse_tool_call(_tool_response("workspace", {"path": "docs"}))

    assert "workspace" in tool_names
    assert parsed["type"] == "workspace_op"
    assert parsed["path"] == "docs"


def test_register_tool_rejects_schema_name_mismatch() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    mismatched_schema = {
        "type": "function",
        "function": {
            "name": "other_name",
            "description": "bad schema",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    try:
        core.register_tool(
            ToolRegistration(
                name="workspace",
                schema=mismatched_schema,
                enabled=lambda _core: True,
                mapper=lambda _core, _args: {"type": "noop"},
            )
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_tool_enabled_map_uses_registry_callback_for_overridden_builtin() -> None:
    core = CognitiveCore(Settings(use_openrouter=False, execution_profile="local_gui"), _DummyComputer())
    core.register_tool(
        ToolRegistration(
            name="browser",
            schema=BROWSER_TOOL,
            enabled=lambda _core: False,
            mapper=lambda _core, args: {"type": "noop", "reason": f"browser disabled: {args}"},
        )
    )

    status = core._tool_enabled_map()
    tool_names = [tool["function"]["name"] for tool in core._available_tools()]

    assert status["browser"] is False
    assert "browser" not in tool_names


def test_system_prompt_uses_registry_status_for_builtin_overrides() -> None:
    core = CognitiveCore(Settings(use_openrouter=False, execution_profile="local_gui"), _DummyComputer())
    core.register_tool(
        ToolRegistration(
            name="browser",
            schema=BROWSER_TOOL,
            enabled=lambda _core: False,
            mapper=lambda _core, args: {"type": "noop", "reason": f"browser disabled: {args}"},
        )
    )
    core.client = MagicMock()
    core.client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    core._call_openrouter(
        observation_b64="",
        history=[],
        include_visual_context=False,
        user_prompt="inspect",
        repeat_info=None,
        plan=None,
        current_step=None,
        loop_state=None,
        ax_tree=None,
        som_tags=None,
        relevant_skills=None,
    )

    call_kwargs = core.client.chat.completions.create.call_args[1]
    system_prompt = call_kwargs["messages"][0]["content"]
    assert "Browser tool is disabled in this execution profile" in system_prompt
