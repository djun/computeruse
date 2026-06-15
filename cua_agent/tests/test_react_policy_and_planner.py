import json

from cua_agent.agent.state_manager import StateManager
from cua_agent.orchestrator.action_policy import ActionPolicy
from cua_agent.orchestrator.planner_client import PlannerClient
from cua_agent.orchestrator.react_types import GroundingBundle
from cua_agent.utils.config import Settings


def test_action_policy_annotates_low_confidence_target_without_blocking() -> None:
    settings = Settings(use_openrouter=False, min_grounding_confidence=0.75)
    policy = ActionPolicy(settings)
    grounding = GroundingBundle(
        screenshot_b64="",
        frame_hash="hash",
        som_tags=[
            {
                "id": 3,
                "gid": "fused:button:submit:1:abc123",
                "confidence": 0.5,
                "frame": {"x": 10, "y": 20, "w": 30, "h": 10},
                "role": "button",
                "label": "Submit",
            }
        ],
    )

    decision = policy.normalize_and_guard(
        {"type": "click_element", "element_id": 3},
        grounding=grounding,
        state=StateManager(),
    )

    assert decision.allowed is True
    assert decision.target_gid == "fused:button:submit:1:abc123"
    assert decision.action["target_gid"] == "fused:button:submit:1:abc123"
    assert decision.action["needs_fresh_grounding"] is True


def test_planner_repairs_invariants_and_fills_react_fields() -> None:
    planner = PlannerClient(Settings(use_openrouter=False), platform_name="Windows")
    raw = {
        "id": "plan",
        "user_prompt": "do it",
        "current_step_index": 0,
        "steps": [
            {
                "id": 0,
                "description": "Open app",
                "success_criteria": "App visible",
                "status": "pending",
                "expected_state": "",
                "recovery_steps": ["Retry"],
                "sub_steps": ["Click app"],
                "preferred_sensor": "vision_full",
                "risk_level": "medium",
                "grounding_strategy": "fusion_required",
            },
            {
                "id": 1,
                "description": "Done",
                "success_criteria": "Done visible",
                "status": "in_progress",
                "expected_state": "text_exists:Done",
                "recovery_steps": [],
                "sub_steps": [],
            },
        ],
    }

    parsed = planner._parse_plan_response(json.dumps(raw), "fallback", "do it")

    assert parsed["current_step_index"] == 1
    assert parsed["steps"][0]["status"] == "done"
    assert parsed["steps"][0]["expected_state"] == "App visible"
    assert parsed["steps"][0]["preferred_sensor"] == "vision_full"
    assert parsed["steps"][0]["risk_level"] == "medium"
    assert parsed["steps"][0]["grounding_strategy"] == "fusion_required"
