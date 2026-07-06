"""Context assembly: one primary UI representation per turn, ranked top-K
candidates, deduplicated history, plan slice, one-line skills, block
instrumentation, and deterministic history compression.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.computer.types import DisplayInfo
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.orchestrator.planning import Plan, Step
from cua_agent.orchestrator.react_types import GroundedNode, GroundingBundle
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


def _core(prefer_native: bool = True) -> CognitiveCore:
    settings = Settings(use_openrouter=False)
    settings.prefer_native_coordinates = prefer_native
    core = CognitiveCore(settings, _DummyComputer())
    core.client = MagicMock()
    core.client.chat.completions.create.return_value = SimpleNamespace(choices=[], usage=None)
    return core


def _node(gid: str, label: str, confidence: float = 0.8) -> GroundedNode:
    return GroundedNode(
        gid=gid,
        role="button",
        label=label,
        path="",
        frame={"x": 1.0, "y": 2.0, "w": 10.0, "h": 10.0},
        source="fused",
        confidence=confidence,
    )


def _grounding(labels: list[str]) -> GroundingBundle:
    return GroundingBundle(
        screenshot_b64="ZnJhbWU=",
        frame_hash="hash",
        fused_nodes=[_node(f"fused:{i}", label) for i, label in enumerate(labels)],
        som_tags=[
            {"id": i, "gid": f"fused:{i}", "label": label, "role": "button", "confidence": 0.8}
            for i, label in enumerate(labels)
        ],
    )


_AX_TREE = {"role": "AXWindow", "title": "Main", "children": []}


def _call(core: CognitiveCore, **overrides) -> str:
    kwargs = dict(
        observation_b64="ZnJhbWU=",
        history=["event_one"],
        include_visual_context=True,
        user_prompt="click the search field",
        repeat_info=None,
        plan=None,
        current_step=None,
        loop_state=None,
        ax_tree=None,
        som_tags=None,
        relevant_skills=None,
        grounding=None,
        state_view=None,
    )
    kwargs.update(overrides)
    core._call_openrouter(**kwargs)
    messages = core.client.chat.completions.create.call_args.kwargs["messages"]
    return messages[0]["content"]


# ---------------------------------------------------------------------------
# One primary UI representation per turn
# ---------------------------------------------------------------------------


def test_native_mode_sends_only_fused_candidates() -> None:
    grounding = _grounding(["Search", "Cancel"])
    prompt = _call(
        _core(prefer_native=True),
        grounding=grounding,
        som_tags=grounding.som_tags,
        ax_tree=_AX_TREE,
    )
    assert "Fused grounding candidates" in prompt
    assert "Detected UI elements" not in prompt
    assert "Numbered overlay marks" not in prompt
    assert "Visible UI Semantic Structure" not in prompt


def test_overlay_mode_keeps_som_as_primary_reference_space() -> None:
    grounding = _grounding(["Search", "Cancel"])
    prompt = _call(
        _core(prefer_native=False),
        grounding=grounding,
        som_tags=grounding.som_tags,
        ax_tree=_AX_TREE,
    )
    assert "Numbered overlay marks" in prompt
    assert "Fused grounding candidates" not in prompt
    assert "Visible UI Semantic Structure" not in prompt


def test_ax_tree_is_fallback_when_no_candidates() -> None:
    prompt = _call(_core(), ax_tree=_AX_TREE)
    assert "Visible UI Semantic Structure" in prompt


def test_text_only_turn_adds_ax_tree_alongside_candidates() -> None:
    grounding = _grounding(["Search"])
    prompt = _call(
        _core(prefer_native=True),
        include_visual_context=False,
        grounding=grounding,
        som_tags=grounding.som_tags,
        ax_tree=_AX_TREE,
    )
    assert "Fused grounding candidates" in prompt
    assert "Visible UI Semantic Structure" in prompt


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------


def test_rank_ui_candidates_prefers_step_keywords_and_caps_at_limit() -> None:
    nodes = [{"label": f"Item {i}", "role": "button", "confidence": 0.5} for i in range(30)]
    nodes.append({"label": "Search field", "role": "textfield", "confidence": 0.4})
    ranked = CognitiveCore._rank_ui_candidates(nodes, "click the search field")
    assert len(ranked) == 12
    assert ranked[0]["label"] == "Search field"


def test_rank_ui_candidates_falls_back_to_confidence() -> None:
    nodes = [
        {"label": "A", "confidence": 0.2},
        {"label": "B", "confidence": 0.9},
    ]
    ranked = CognitiveCore._rank_ui_candidates(nodes, "unrelated query text")
    assert ranked[0]["label"] == "B"


def test_prompt_candidates_capped_at_twelve() -> None:
    grounding = _grounding([f"Button {i}" for i in range(40)])
    prompt = _call(_core(prefer_native=True), grounding=grounding)
    candidate_lines = [line for line in prompt.splitlines() if line.strip().startswith("- fused:")]
    assert len(candidate_lines) == 12


# ---------------------------------------------------------------------------
# History deduplication and plan slice
# ---------------------------------------------------------------------------


def test_state_view_drops_recent_history_duplicate() -> None:
    prompt = _call(
        _core(),
        state_view={"steps": 3, "recent_history": ["dup_line_marker"]},
    )
    assert "Typed state view" in prompt
    assert "dup_line_marker" not in prompt
    assert "recent_history" not in prompt


def test_plan_slice_shows_only_current_and_next_step() -> None:
    steps = [
        Step(id=0, description="step zero", success_criteria="zero done", status="done"),
        Step(id=1, description="step one", success_criteria="one done", status="in_progress"),
        Step(id=2, description="step two", success_criteria="two done"),
        Step(id=3, description="step three", success_criteria="three done"),
    ]
    plan = Plan(id="p", user_prompt="task", steps=steps, current_step_index=1)
    prompt = _call(_core(), plan=plan, current_step=steps[1])
    assert "step one" in prompt
    assert "Next step (context only): step two" in prompt
    assert "step three" not in prompt
    assert "step zero" not in prompt


# ---------------------------------------------------------------------------
# Skills rendering
# ---------------------------------------------------------------------------


def test_skills_rendered_as_single_lines_capped_at_three() -> None:
    skills = [
        SimpleNamespace(
            name=f"skill{i}",
            id=f"id{i}",
            description="d" * 300,
            parameters={"query": {"required": True, "description": "the query"}},
        )
        for i in range(5)
    ]
    prompt = _call(_core(), relevant_skills=skills)
    assert "skill0 (ID: id0) args=[query]" in prompt
    assert "skill2" in prompt
    assert "skill3" not in prompt
    # Long description truncated to 80 chars; the raw 300-char blob must not appear.
    assert "d" * 100 not in prompt


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_context_report_estimates_blocks_and_image_flag() -> None:
    core = _core()
    grounding = _grounding(["Search"])
    _call(core, grounding=grounding, state_view={"steps": 1})
    report = core.last_context_report
    for key in ("plan", "skills", "candidates", "state_view", "history", "system_total", "image"):
        assert key in report
    assert report["image"] is True
    assert report["system_total"] > 0
    assert report["candidates"] > 0


# ---------------------------------------------------------------------------
# Deterministic history compression
# ---------------------------------------------------------------------------


def _orch_for_compression() -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.logger = MagicMock()
    orch.planner = MagicMock()
    return orch


def test_compress_history_is_deterministic_and_skips_planner() -> None:
    from cua_agent.agent.state_manager import StateManager

    orch = _orch_for_compression()
    state = StateManager()
    state.history = ["user_prompt:do it"]
    state.history += [
        "action:{'type': 'left_click', 'success': True, 'reason': 'ok'}",
        "action:{'type': 'left_click', 'success': False, 'reason': 'missed'}",
        "verification_contract_failed:expected='x'",
        "observation@1:changed=True",
        "plan_step_completed:0",
    ] * 4
    state.history += [f"filler_{i}" for i in range(50)]

    orch._compress_history(state)

    assert state.history[1].startswith("history_summary:")
    summary = state.history[1]
    assert "left_clickx8" in summary
    assert "4 verification failures" in summary
    assert "plan_step_completed" in summary
    orch.planner.summarize_history_chunk.assert_not_called()


def test_compact_history_chunk_handles_unstructured_lines() -> None:
    summary = Orchestrator._compact_history_chunk(["free text", "another line"])
    assert summary == "2 events"


# ---------------------------------------------------------------------------
# Skill prompt threshold
# ---------------------------------------------------------------------------


def test_skill_prompt_threshold_is_softer_than_fast_path() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    orch.settings = Settings(use_openrouter=False)
    vector_threshold = orch._skill_prompt_threshold("vector")
    keyword_threshold = orch._skill_prompt_threshold("keyword")
    assert vector_threshold < float(orch.settings.fast_path_min_vector_score)
    assert keyword_threshold < float(orch.settings.fast_path_min_keyword_score)
    assert orch._skill_prompt_threshold("chroma") == vector_threshold
