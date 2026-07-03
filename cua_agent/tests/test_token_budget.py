from types import SimpleNamespace
from unittest.mock import MagicMock

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import DisplayInfo
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.utils.config import Settings
from cua_agent.utils.token_usage import usage_tokens


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


def test_usage_tokens_reads_total_or_defaults_to_zero() -> None:
    assert usage_tokens(SimpleNamespace(usage=SimpleNamespace(total_tokens=42))) == 42
    assert usage_tokens(SimpleNamespace(usage={"total_tokens": 7})) == 7
    assert usage_tokens(SimpleNamespace(usage=None)) == 0
    assert usage_tokens(SimpleNamespace()) == 0
    assert usage_tokens(SimpleNamespace(usage=SimpleNamespace(total_tokens=None))) == 0


def test_cognitive_core_accumulates_token_usage() -> None:
    core = CognitiveCore(Settings(use_openrouter=False), _DummyComputer())
    core.client = MagicMock()
    core.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[], usage=SimpleNamespace(total_tokens=123)
    )

    assert core.tokens_used == 0
    core._call_openrouter(
        observation_b64="",
        history=[],
        include_visual_context=False,
        user_prompt="x",
        repeat_info=None,
        plan=None,
        current_step=None,
        loop_state=None,
        ax_tree=None,
        som_tags=None,
        relevant_skills=None,
    )
    assert core.tokens_used == 123
    core._call_openrouter(
        observation_b64="",
        history=[],
        include_visual_context=False,
        user_prompt="x",
        repeat_info=None,
        plan=None,
        current_step=None,
        loop_state=None,
        ax_tree=None,
        som_tags=None,
        relevant_skills=None,
    )
    assert core.tokens_used == 246


def test_orchestrator_total_tokens_sums_components() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = SimpleNamespace(tokens_used=100)
    orch.cognitive_core = SimpleNamespace(tokens_used=250)
    orch.reflector = SimpleNamespace(tokens_used=50)
    assert orch._total_tokens_used() == 400
