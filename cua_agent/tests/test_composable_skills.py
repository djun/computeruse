import logging

from cua_agent.memory.skill_store import ProceduralSkill, SkillStore
from cua_agent.orchestrator.orchestrator import Orchestrator


def _orchestrator() -> Orchestrator:
    return Orchestrator.__new__(Orchestrator)


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
    orchestrator = _orchestrator()
    skill = _skill()

    actions, resolved_args, missing = orchestrator._materialize_skill_actions(
        skill,
        {"query_text": "teclado mecanico"},
    )

    assert missing == []
    assert resolved_args["query_text"] == "teclado mecanico"
    assert actions[1]["text"] == "teclado mecanico"

    contract = orchestrator._render_skill_verification_contract(skill, resolved_args)
    assert contract["expected_state"] == "text_exists:teclado mecanico"


def test_materialize_skill_actions_requires_missing_required_args() -> None:
    orchestrator = _orchestrator()
    skill = _skill()

    actions, _, missing = orchestrator._materialize_skill_actions(skill, {})

    assert actions[1]["text"] == "{query_text}"
    assert "query_text" in missing


def test_build_composable_skill_payload_creates_parameter_templates() -> None:
    orchestrator = _orchestrator()

    templated, parameters = orchestrator._build_composable_skill_payload(
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
