import json
import unittest
from unittest.mock import MagicMock, patch

from macos_cua_agent.orchestrator.planning import Step, Plan
from macos_cua_agent.orchestrator.planner_client import PlannerClient
from macos_cua_agent.orchestrator.reflection import Reflector, ReflectionResult
from macos_cua_agent.utils.config import Settings

class TestArchitecture(unittest.TestCase):
    
    def setUp(self):
        self.settings = Settings()
        # Disable real API calls
        self.settings.planner_api_key = "mock_key"
        self.settings.reflector_api_key = "mock_key"
        self.settings.enable_reflection = True

    def test_plan_parsing_recovery_steps(self):
        """Verify recovery_steps are parsed correctly."""
        client = PlannerClient(self.settings)
        client.client = MagicMock() # Mock OpenAI client
        
        raw_json = {
            "id": "test_plan",
            "user_prompt": "do something",
            "steps": [
                {
                    "id": 1,
                    "description": "Step 1",
                    "success_criteria": "Done",
                    "status": "pending",
                    "recovery_steps": ["Retry A", "Retry B"]
                }
            ]
        }
        
        parsed = client._parse_plan_response(json.dumps(raw_json), "test_plan", "do something")
        plan = Plan.from_dict(parsed)
        
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].recovery_steps, ["Retry A", "Retry B"])

    @patch('macos_cua_agent.orchestrator.reflection.Reflector._build_client')
    def test_reflector_structured_response(self, mock_build):
        """Verify Reflector returns structured result and uses correct schema params."""
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        
        reflector = Reflector(self.settings)
        reflector.client = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "is_complete": False,
            "status": "failed",
            "failure_type": "visual_mismatch",
            "reason": "Button not found"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        step = Step(id=1, description="Click button", success_criteria="Button gone")
        result = reflector.evaluate_step(step, [], "base64_img", True)
        
        self.assertIsInstance(result, ReflectionResult)
        self.assertFalse(result.is_complete)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_type, "visual_mismatch")
        self.assertEqual(result.reason, "Button not found")

        # Verify the call arguments for structured output
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertIn("response_format", call_kwargs)
        self.assertEqual(call_kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(call_kwargs["response_format"]["json_schema"]["name"], "reflection_result")
        self.assertTrue(call_kwargs["response_format"]["json_schema"]["strict"])

    @patch('macos_cua_agent.orchestrator.planner_client.PlannerClient._build_client')
    def test_planner_uses_structured_outputs(self, mock_build):
        """Planner should request structured JSON responses."""
        mock_client = MagicMock()
        mock_build.return_value = mock_client

        planner = PlannerClient(self.settings)
        planner.client = mock_client

        message = MagicMock()
        message.parsed = {
            "id": "plan_123",
            "user_prompt": "do something",
            "steps": [
                {"id": 0, "description": "Step 1", "success_criteria": "Seen", "status": "pending"}
            ],
            "current_step_index": 0,
        }
        message.content = message.parsed
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=message)]
        mock_client.chat.completions.create.return_value = fake_response

        planner.make_plan("do something")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        response_format = call_kwargs.get("response_format", {})
        self.assertEqual(response_format.get("type"), "json_schema")
        self.assertTrue(response_format.get("json_schema", {}).get("strict"))
        self.assertIn("structured_outputs", call_kwargs.get("extra_body", {}))

    def test_parse_plan_response_uses_parsed_payload(self):
        """Ensure parsed structured outputs are consumed directly."""
        planner = PlannerClient(self.settings)
        message = MagicMock()
        message.parsed = {
            "id": "plan_abc",
            "user_prompt": "sample task",
            "steps": [
                {"id": 0, "description": "Step 0", "success_criteria": "Done", "status": "pending"}
            ],
            "current_step_index": 1,
        }
        message.content = message.parsed

        parsed = planner._parse_plan_response(message, "fallback", "sample task")
        self.assertEqual(parsed["id"], "plan_abc")
        self.assertEqual(parsed["user_prompt"], "sample task")
        self.assertEqual(parsed["current_step_index"], 1)
if __name__ == '__main__':
    unittest.main()
