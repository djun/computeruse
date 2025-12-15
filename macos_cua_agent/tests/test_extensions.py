import unittest
import tempfile
import os
import subprocess
import uuid
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import ActionResult, StateManager
from cua_agent.computer.types import DisplayInfo
from macos_cua_agent.drivers.action_engine import ActionEngine
from macos_cua_agent.drivers.browser_driver import BrowserDriver
from cua_agent.memory.memory_manager import MemoryManager
from cua_agent.policies.policy_engine import PolicyEngine, PolicyDecision
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

    def run_health_checks(self, settings: Settings, logger=None) -> None:  # noqa: ARG002
        return None

class TestExtensions(unittest.TestCase):

    def setUp(self):
        self.settings = Settings()
        # Mocking dependencies for CognitiveCore if needed, but we are testing a pure method
        self.core = CognitiveCore(self.settings, _DummyComputer())
        # Create a PolicyEngine with a dummy rules file (will use defaults + overrides)
        self.policy = PolicyEngine("dummy_rules.yaml", self.settings)

    def test_map_drag_and_drop(self):
        args = {
            "action": "drag_and_drop",
            "x": 100, "y": 100,
            "target_x": 200, "target_y": 200,
            "duration": 2.0,
            "hold_delay": 0.5
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "drag_and_drop")
        self.assertEqual(result["x"], 100)
        self.assertEqual(result["target_x"], 200)
        self.assertEqual(result["duration"], 2.0)
        self.assertEqual(result["hold_delay"], 0.5)

    def test_map_hover(self):
        args = {
            "action": "hover",
            "x": 50, "y": 50,
            "duration": 1.5
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "hover")
        self.assertEqual(result["duration"], 1.5)
    
    def test_map_select_area(self):
        args = {
            "action": "select_area",
            "x": 10, "y": 10,
            "target_x": 20, "target_y": 30,
            "duration": 0.6
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "select_area")
        self.assertEqual(result["x"], 10)
        self.assertEqual(result["target_y"], 30)
        self.assertEqual(result["duration"], 0.6)

    def test_map_probe_ui(self):
        args = {
            "action": "probe_ui",
            "x": 10, "y": 20,
            "radius": 15
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "probe_ui")
        self.assertEqual(result["x"], 10)
        self.assertEqual(result["radius"], 15.0)

    def test_map_clipboard_op(self):
        args = {
            "action": "clipboard_op",
            "sub_action": "write",
            "content": "secret"
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "clipboard_op")
        self.assertEqual(result["sub_action"], "write")
        self.assertEqual(result["content"], "secret")
    
    def test_map_verify_after_flag(self):
        args = {"action": "left_click", "x": 1, "y": 2, "verify_after": False}
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "left_click")
        self.assertFalse(result["verify_after"])

    def test_map_phantom_mode(self):
        args = {
            "action": "left_click",
            "x": 10, "y": 10,
            "phantom_mode": True
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "left_click")
        self.assertTrue(result["phantom_mode"])
    
    def test_map_run_skill(self):
        args = {"action": "run_skill", "skill_name": "fill-form", "verify_after": False}
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "run_skill")
        self.assertEqual(result["skill_name"], "fill-form")
        self.assertFalse(result["verify_after"])
    
    def test_phantom_mode_right_click_action_engine(self):
        engine = ActionEngine(self.settings, self.policy)
        engine.accessibility_driver.perform_action_at = MagicMock(return_value=ActionResult(True, "ax"))
        engine.hid_driver.right_click = MagicMock(return_value=ActionResult(True, "hid"))
        action = {"type": "right_click", "x": 5, "y": 5, "phantom_mode": True}

        result = engine.execute(action)
        self.assertTrue(result.success)
        engine.hid_driver.right_click.assert_not_called()

    def test_phantom_auto_with_element_id(self):
        engine = ActionEngine(self.settings, self.policy)
        engine.accessibility_driver.perform_action_at = MagicMock(return_value=ActionResult(True, "ax"))
        engine.hid_driver.left_click = MagicMock(return_value=ActionResult(True, "hid"))

        action = {"type": "left_click", "x": 1, "y": 2, "element_id": 42}
        result = engine.execute(action)

        self.assertTrue(result.success)
        engine.accessibility_driver.perform_action_at.assert_called_once()
        engine.hid_driver.left_click.assert_not_called()

    @patch("subprocess.check_output")
    def test_clipboard_redaction(self, mock_paste):
        mock_paste.return_value = b"AKIA" + b"1" * 16  # Looks like AWS access key -> should redact
        engine = ActionEngine(self.settings, self.policy)

        result = engine.execute({"type": "clipboard_op", "sub_action": "read"})

        self.assertTrue(result.success)
        self.assertTrue(result.metadata.get("redacted"))
        self.assertEqual(result.metadata.get("content"), "<REDACTED>")
    
    def test_skill_store_dedup(self):
        tmp_root = Path(os.getcwd()) / f".tmp_memory_{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True, exist_ok=False)
            settings = Settings(memory_root=str(tmp_root))
            memory = MemoryManager(settings)
            actions = [{"type": "left_click", "x": 1, "y": 2}]

            skill1 = memory.save_skill("click-once", "click action", actions)
            skill2 = memory.save_skill("click-repeat", "another desc", actions)

            self.assertEqual(skill1.id, skill2.id)
            self.assertEqual(len(memory.list_skills()), 1)
            self.assertGreaterEqual(skill2.usage_count, 1)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_policy_exclusion_zone(self):
        # Inject a rule manually
        self.policy.rules["exclusion_zones"] = [
            {"x": 0, "y": 0, "w": 100, "h": 100, "label": "TopLeftCorner"}
        ]
        
        # Allowed action (outside zone)
        allowed = self.policy.evaluate({"type": "left_click", "x": 150, "y": 150})
        self.assertTrue(allowed.allowed)

        # Blocked action (inside zone)
        blocked = self.policy.evaluate({"type": "left_click", "x": 50, "y": 50})
        self.assertFalse(blocked.allowed)
        self.assertIn("exclusion zone", blocked.reason)

        # Blocked drag start
        blocked_drag = self.policy.evaluate({"type": "drag_and_drop", "x": 50, "y": 50, "target_x": 200, "target_y": 200})
        self.assertFalse(blocked_drag.allowed)

        # Blocked drag end
        blocked_drag_end = self.policy.evaluate({"type": "drag_and_drop", "x": 200, "y": 200, "target_x": 50, "target_y": 50})
        self.assertFalse(blocked_drag_end.allowed)

    def test_policy_run_javascript_requires_hitl(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("hitl_actions:\n  - run_javascript\n")
            tmp.flush()
            rules_path = tmp.name
        try:
            policy = PolicyEngine(rules_path, self.settings)
            decision = policy.evaluate({"type": "browser_op", "command": "run_javascript"})
            self.assertTrue(decision.allowed)
            self.assertTrue(decision.hitl_required)
        finally:
            os.remove(rules_path)

    def test_default_policy_run_javascript_hitl(self):
        policy = PolicyEngine("nonexistent_rules.yaml", self.settings)
        decision = policy.evaluate({"type": "browser_op", "command": "run_javascript"})
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.hitl_required)

    def test_browser_shadow_dom_payload(self):
        driver = BrowserDriver(self.settings)
        with patch.object(driver, "_run_js_with_result", return_value=ActionResult(True, "ok")) as mock_run:
            driver._get_dom_tree("Safari")
            called_js = mock_run.call_args[0][1]
            # Ensure shadow DOM traversal is present in payload
            self.assertIn("shadowRoot", called_js)
            self.assertIn("#shadow-root", called_js)

    def test_browser_run_js_promise_wait_wrapper(self):
        driver = BrowserDriver(self.settings)
        with patch.object(driver, "_run_js_with_result", return_value=ActionResult(True, "ok")) as mock_run:
            driver._run_arbitrary_js("Safari", "return Promise.resolve(1);")
            called_js = mock_run.call_args[0][1]
            # Promise wait logic should be embedded to avoid missing-value returns
            self.assertIn("Promise unresolved", called_js)
            self.assertIn("runner.then", called_js)

    def test_browser_result_added_to_history(self):
        state = StateManager()
        action = {"type": "browser_op", "execution": "browser", "command": "get_page_content"}
        result = ActionResult(success=True, reason="ok", metadata={"data": {"status": "success", "result": "Hello Web"}})

        state.record_action(action, result)

        browser_entries = [h for h in state.history if h.startswith("browser_result")]
        self.assertEqual(len(browser_entries), 1)
        self.assertIn("get_page_content", browser_entries[0])
        self.assertIn("Hello Web", browser_entries[0])

    @patch("macos_cua_agent.drivers.browser_driver.subprocess.run")
    def test_applescript_timeout_is_handled(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["osascript"], timeout=0.1)
        driver = BrowserDriver(self.settings)

        result = driver._run_arg_applescript("Safari", "return \"ok\"", [], "timeout_test", timeout=0.1)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.reason)

if __name__ == '__main__':
    unittest.main()
