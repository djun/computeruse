import logging

from cua_agent.memory.skill_composer import SkillComposer
from cua_agent.memory.skill_store import ProceduralSkill, SkillStore


def _composer() -> SkillComposer:
    return SkillComposer()


def _skill() -> ProceduralSkill:
    return ProceduralSkill(
        id="skill-1",
        name="web_search_bar_interaction",
        description="Foca na barra de pesquisa principal e envia query.",
        actions=[
            {"type": "left_click", "semantic_role": "searchbox"},
            {"type": "type", "text": "{query_text}"},
            {"type": "key", "keys": ["enter"]},
        ],
        created_at=0.0,
        updated_at=0.0,
        parameters={
            "query_text": {
                "description": "Termo da pesquisa",
                "required": True,
            }
        },
        verification_contract={
            "sensor": "a11y_tree",
            "expected_state": "text_exists:{query_text}",
            "timeout_seconds": 5,
        },
    )


def test_skill_store_persists_parameters_and_verification_contract(tmp_path) -> None:
    store = SkillStore(tmp_path, logging.getLogger("test"))
    skill = store.save_skill(
        name="web_search_bar_interaction",
        description="Foca na barra de pesquisa principal e envia query.",
        actions=[
            {"type": "left_click", "semantic_role": "searchbox"},
            {"type": "type", "text": "{query_text}"},
            {"type": "key", "keys": ["enter"]},
        ],
        parameters={
            "query_text": {
                "description": "Termo da pesquisa",
                "required": True,
            }
        },
        verification_contract={
            "sensor": "a11y_tree",
            "expected_state": "text_exists:{query_text}",
            "timeout_seconds": 5,
        },
    )

    loaded = store.get_skill(skill.id)
    assert loaded is not None
    assert loaded.parameters["query_text"]["required"] is True
    assert loaded.verification_contract["sensor"] == "a11y_tree"
    assert loaded.actions[1]["text"] == "{query_text}"


def test_materialize_skill_actions_applies_runtime_args() -> None:
    composer = _composer()
    skill = _skill()

    actions, resolved_args, missing = composer._materialize_skill_actions(
        skill,
        {"query_text": "teclado mecanico"},
    )

    assert missing == []
    assert resolved_args["query_text"] == "teclado mecanico"
    assert actions[1]["text"] == "teclado mecanico"

    contract = composer._render_skill_verification_contract(skill, resolved_args)
    assert contract["expected_state"] == "text_exists:teclado mecanico"


def test_materialize_skill_actions_requires_missing_required_args() -> None:
    composer = _composer()
    skill = _skill()

    actions, _, missing = composer._materialize_skill_actions(skill, {})

    assert actions[1]["text"] == "{query_text}"
    assert "query_text" in missing


def test_skill_context_metadata_stamps_platform_precondition() -> None:
    actions = [{"type": "left_click", "semantic_label": "Submit", "semantic_role": "AXButton"}]
    pre, sig = SkillComposer(platform_name="macOS")._skill_context_metadata(actions)
    assert pre["platform"] == "macOS"
    assert "Submit" in sig["labels"]


def test_skill_context_metadata_omits_platform_when_unknown() -> None:
    # Blank platform => no precondition => the fast-path filter stays platform-agnostic
    # (regression: previously stamped "unknown" and rejected every learned skill).
    pre, _ = SkillComposer()._skill_context_metadata([{"type": "left_click"}])
    assert "platform" not in pre


def test_build_composable_skill_payload_creates_parameter_templates() -> None:
    composer = _composer()

    templated, parameters = composer._build_composable_skill_payload(
        [
            {"type": "open_app", "app_name": "Google Chrome"},
            {"type": "type", "text": "Relatorio financeiro janeiro"},
            {"type": "key", "keys": ["enter"]},
        ]
    )

    assert templated[0]["app_name"] == "{app_name_1}"
    assert templated[1]["text"] == "{text_1}"
    assert parameters["app_name_1"]["default"] == "Google Chrome"
    assert parameters["text_1"]["default"] == "Relatorio financeiro janeiro"
