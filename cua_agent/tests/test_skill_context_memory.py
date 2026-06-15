from cua_agent.memory.memory_manager import MemoryManager
from cua_agent.memory.skill_store import SkillStore
from cua_agent.utils.config import Settings


def test_skill_fingerprint_includes_preconditions(tmp_path) -> None:
    store = SkillStore(tmp_path, logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})())
    actions = [{"type": "key", "keys": ["ctrl", "l"]}, {"type": "type", "text": "{url}"}]

    first = store.save_skill(
        name="focus-url",
        description="Focus URL field",
        actions=actions,
        preconditions={"platform": "Windows"},
    )
    second = store.save_skill(
        name="focus-url",
        description="Focus URL field",
        actions=actions,
        preconditions={"platform": "macOS"},
    )

    assert first.id != second.id
    assert len(store.list_skills()) == 2


def test_fast_path_skips_skill_when_preconditions_do_not_match(tmp_path) -> None:
    settings = Settings(use_openrouter=False, memory_root=str(tmp_path))
    memory = MemoryManager(settings)
    memory.save_skill(
        name="focus-url",
        description="Focus URL field",
        actions=[{"type": "key", "keys": ["ctrl", "l"]}],
        tags=["browser"],
        preconditions={"platform": "Windows"},
    )

    skipped = memory.select_fast_path_skill(
        "focus url field",
        min_keyword_score=1.0,
        context={"platform": "macOS"},
        grounding_signature={},
    )
    matched = memory.select_fast_path_skill(
        "focus url field",
        min_keyword_score=1.0,
        context={"platform": "Windows"},
        grounding_signature={},
    )

    assert skipped is None
    assert matched is not None
