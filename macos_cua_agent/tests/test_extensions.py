import unittest
import tempfile
import os
import subprocess
import uuid
import base64
import io
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
from PIL import Image
from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import ActionResult, StateManager, VerificationContract
from cua_agent.computer.types import DisplayInfo
from macos_cua_agent.drivers.action_engine import ActionEngine
from macos_cua_agent.drivers.browser_driver import BrowserDriver
from macos_cua_agent.drivers.shell_driver import ShellDriver
from macos_cua_agent.drivers.vision_pipeline import VisionPipeline
from cua_agent.memory.memory_manager import MemoryManager
from cua_agent.orchestrator.orchestrator import Orchestrator
from cua_agent.orchestrator.planning import Step
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

    def _make_engine(self) -> ActionEngine:
        """
        Build an ActionEngine without touching real macOS UI frameworks.

        In some sandbox/CI environments, importing or calling AppKit/Quartz can abort the
        interpreter. These tests only exercise ActionEngine routing logic, so we stub
        out the OS-facing drivers.
        """
        dummy_display = DisplayInfo(
            logical_width=1280,
            logical_height=720,
            physical_width=1280,
            physical_height=720,
            scale_factor=1.0,
        )
        with (
            patch("macos_cua_agent.drivers.action_engine.get_display_info", return_value=dummy_display),
            patch("macos_cua_agent.drivers.action_engine.HIDDriver", return_value=MagicMock()),
            patch("macos_cua_agent.drivers.action_engine.SemanticDriver", return_value=MagicMock()),
            patch("macos_cua_agent.drivers.action_engine.ShellDriver", return_value=MagicMock()),
            patch("macos_cua_agent.drivers.action_engine.AccessibilityDriver", return_value=MagicMock()),
            patch("macos_cua_agent.drivers.action_engine.BrowserDriver", return_value=MagicMock()),
        ):
            return ActionEngine(self.settings, self.policy)

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

    def test_map_verification_contract(self):
        args = {
            "action": "left_click",
            "x": 10,
            "y": 12,
            "verification": {
                "sensor": "a11y_tree",
                "expected_state": "text_exists:Dashboard",
                "timeout_seconds": 7,
            },
        }
        result = self.core._map_single_computer_action(args)
        self.assertEqual(result["type"], "left_click")
        self.assertEqual(result["verification"]["sensor"], "a11y_tree")
        self.assertEqual(result["verification"]["expected_state"], "text_exists:Dashboard")
        self.assertEqual(result["verification"]["timeout_seconds"], 7)

    def test_map_macro_verification_contract(self):
        args = {
            "actions": [{"action": "left_click", "x": 5, "y": 6}],
            "verification": {
                "sensor": "os_telemetry",
                "expected_state": "clipboard_changed",
                "timeout_seconds": 4,
            },
        }
        result = self.core._map_tool_args(args)
        self.assertEqual(result["type"], "macro_actions")
        self.assertEqual(result["verification"]["sensor"], "os_telemetry")
        self.assertEqual(result["verification"]["expected_state"], "clipboard_changed")
        self.assertEqual(result["verification"]["timeout_seconds"], 4)

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

    def test_available_tools_local_gui_profile(self):
        core = CognitiveCore(Settings(execution_profile="local_gui"), _DummyComputer())
        names = [tool["function"]["name"] for tool in core._available_tools()]
        self.assertIn("computer", names)
        self.assertIn("browser", names)
        self.assertIn("notebook", names)
        self.assertNotIn("shell", names)

    def test_available_tools_remote_cli_profile_with_shell_enabled(self):
        core = CognitiveCore(Settings(execution_profile="remote_cli", enable_shell=True), _DummyComputer())
        names = [tool["function"]["name"] for tool in core._available_tools()]
        self.assertIn("shell", names)
        self.assertIn("script", names)
        self.assertIn("notebook", names)
        self.assertNotIn("computer", names)
        self.assertNotIn("browser", names)

    def test_available_tools_remote_cli_profile_shell_disabled_by_default(self):
        core = CognitiveCore(Settings(execution_profile="remote_cli"), _DummyComputer())
        names = [tool["function"]["name"] for tool in core._available_tools()]
        self.assertNotIn("shell", names)
        self.assertNotIn("script", names)
        self.assertIn("notebook", names)
        self.assertNotIn("computer", names)
        self.assertNotIn("browser", names)

    def test_profile_blocks_computer_mapping_in_remote_cli(self):
        core = CognitiveCore(Settings(execution_profile="remote_cli"), _DummyComputer())
        result = core._map_tool_args({"action": "left_click", "x": 1, "y": 2})
        self.assertEqual(result["type"], "noop")
        self.assertIn("remote_cli", result["reason"])

    def test_profile_blocks_shell_mapping_in_local_gui(self):
        core = CognitiveCore(Settings(execution_profile="local_gui"), _DummyComputer())
        result = core._map_shell_args({"command": "echo hello"})
        self.assertEqual(result["type"], "noop")
        self.assertIn("local_gui", result["reason"])

    def test_shell_mapping_blocked_when_shell_flag_disabled(self):
        core = CognitiveCore(Settings(execution_profile="remote_cli", enable_shell=False), _DummyComputer())
        result = core._map_shell_args({"command": "echo hello"})
        self.assertEqual(result["type"], "noop")
        self.assertIn("ENABLE_SHELL=false", result["reason"])

    def test_cognitive_core_text_only_turn_omits_image_payload(self):
        core = CognitiveCore(Settings(use_openrouter=True), _DummyComputer())
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        core.client = mock_client

        core._call_openrouter(
            observation_b64="abc123",
            history=["h1"],
            include_visual_context=False,
            user_prompt="do x",
            repeat_info=None,
            plan=None,
            current_step=None,
            loop_state=None,
            ax_tree=None,
            som_tags=None,
            relevant_skills=None,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertTrue(any(item.get("type") == "text" for item in user_content))
        self.assertFalse(any(item.get("type") == "image_url" for item in user_content))
    
    def test_phantom_mode_right_click_action_engine(self):
        engine = self._make_engine()
        engine.accessibility_driver.perform_action_at = MagicMock(return_value=ActionResult(True, "ax"))
        engine.hid_driver.right_click = MagicMock(return_value=ActionResult(True, "hid"))
        action = {"type": "right_click", "x": 5, "y": 5, "phantom_mode": True}

        result = engine.execute(action)
        self.assertTrue(result.success)
        engine.hid_driver.right_click.assert_not_called()

    def test_phantom_auto_with_element_id(self):
        engine = self._make_engine()
        engine.accessibility_driver.perform_action_at = MagicMock(return_value=ActionResult(True, "ax"))
        engine.hid_driver.left_click = MagicMock(return_value=ActionResult(True, "hid"))

        action = {"type": "left_click", "x": 1, "y": 2, "element_id": 42}
        result = engine.execute(action)

        self.assertTrue(result.success)
        engine.accessibility_driver.perform_action_at.assert_called_once()
        engine.hid_driver.left_click.assert_not_called()

    def test_execution_profile_remote_cli_blocks_gui_action(self):
        self.settings.execution_profile = "remote_cli"
        engine = self._make_engine()
        result = engine.execute({"type": "left_click", "x": 10, "y": 10})
        self.assertFalse(result.success)
        self.assertIn("remote_cli", result.reason)

    def test_execution_profile_local_gui_blocks_shell_action(self):
        self.settings.execution_profile = "local_gui"
        engine = self._make_engine()
        result = engine.execute({"type": "sandbox_shell", "execution": "shell", "cmd": "echo hello"})
        self.assertFalse(result.success)
        self.assertIn("local_gui", result.reason)

    def test_shell_driver_blocks_when_profile_disallows_shell(self):
        settings = Settings(execution_profile="local_gui", enable_shell=True)
        driver = ShellDriver(settings)
        result = driver.execute({"command": "echo hello"})
        self.assertFalse(result.success)
        self.assertIn("local_gui", result.reason)

    def test_shell_driver_script_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                execution_profile="remote_cli",
                enable_shell=True,
                shell_workspace_root=tmp,
            )
            driver = ShellDriver(settings)

            write_res = driver.execute(
                {
                    "type": "script_op",
                    "operation": "write",
                    "path": "tools/demo.py",
                    "content": "print('hello from script')\n",
                }
            )
            self.assertTrue(write_res.success)

            read_res = driver.execute(
                {
                    "type": "script_op",
                    "operation": "read",
                    "path": "tools/demo.py",
                }
            )
            self.assertTrue(read_res.success)
            self.assertIn("hello from script", read_res.metadata.get("stdout", ""))

    def test_shell_driver_script_run_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                execution_profile="remote_cli",
                enable_shell=True,
                shell_workspace_root=tmp,
            )
            driver = ShellDriver(settings)
            driver.execute(
                {
                    "type": "script_op",
                    "operation": "write",
                    "path": "tools/run_me.py",
                    "content": "print('RUN_OK')\n",
                }
            )

            run_res = driver.execute(
                {
                    "type": "script_op",
                    "operation": "run",
                    "path": "tools/run_me.py",
                }
            )
            self.assertTrue(run_res.success)
            self.assertIn("RUN_OK", run_res.metadata.get("stdout", ""))

    def test_inspect_ui_visual_fallback_when_ax_fails(self):
        engine = self._make_engine()
        engine.accessibility_driver.get_active_window_tree = MagicMock(return_value=ActionResult(False, "ax unavailable"))
        vision = MagicMock()
        vision.capture_base64.return_value = "fake_frame"
        vision.detect_ui_elements.return_value = [
            {
                "role": "AXButton",
                "title": "Submit",
                "label": "Submit",
                "frame": {"x": 10, "y": 20, "w": 80, "h": 30},
                "source": "detector_ultralytics",
            }
        ]
        engine.vision_pipeline = vision

        result = engine.execute({"type": "inspect_ui"})

        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get("grounding"), "vision_fallback")
        self.assertEqual(result.metadata.get("element_count"), 1)

    def test_phantom_click_visual_retarget_fallback(self):
        engine = self._make_engine()
        engine.accessibility_driver.perform_action_at = MagicMock(return_value=ActionResult(False, "no ax target"))
        engine.hid_driver.left_click = MagicMock(return_value=ActionResult(True, "hid click"))
        vision = MagicMock()
        vision.capture_base64.return_value = "fake_frame"
        vision.detect_ui_elements.return_value = [
            {
                "role": "AXButton",
                "title": "Submit",
                "label": "Submit",
                "frame": {"x": 100, "y": 200, "w": 40, "h": 20},
                "source": "ocr",
                "path": "vision.ocr.1",
            }
        ]
        engine.vision_pipeline = vision

        action = {
            "type": "left_click",
            "x": 2,
            "y": 3,
            "element_id": 42,
            "phantom_mode": True,
            "semantic_label": "Submit",
        }
        result = engine.execute(action)

        self.assertTrue(result.success)
        engine.hid_driver.left_click.assert_called_once_with(120.0, 210.0)

    @patch("subprocess.check_output")
    def test_clipboard_redaction(self, mock_paste):
        mock_paste.return_value = b"AKIA" + b"1" * 16  # Looks like AWS access key -> should redact
        engine = self._make_engine()

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

    def test_fast_path_skill_selection_keyword(self):
        tmp_root = Path(os.getcwd()) / f".tmp_memory_{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True, exist_ok=False)
            settings = Settings(memory_root=str(tmp_root), enable_embeddings=False)
            memory = MemoryManager(settings)
            actions = [{"type": "left_click", "x": 1, "y": 2}]
            target = memory.save_skill(
                "cancel-subscription-x",
                "cancel subscription x in billing settings",
                actions,
                tags=["billing", "subscription"],
            )
            memory.save_skill(
                "open-dashboard",
                "open dashboard and view metrics",
                actions,
                tags=["dashboard"],
            )

            match = memory.select_fast_path_skill("cancel subscription x")

            self.assertIsNotNone(match)
            self.assertEqual(match.skill.id, target.id)
            self.assertEqual(match.strategy, "keyword")
            self.assertGreaterEqual(match.score, 4.0)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_dynamic_skill_synthesis_after_recovery(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.settings = Settings(dynamic_skill_min_actions=3)
        orchestrator.logger = MagicMock()
        orchestrator.memory = MagicMock()

        step = Step(id=7, description="Confirm cancellation", success_criteria="Cancellation confirmed")
        trace = [
            {"success": False, "changed": False, "reason": "miss", "action": {"type": "left_click", "x": 10, "y": 10}},
            {
                "success": True,
                "changed": True,
                "reason": "ok",
                "action": {"type": "left_click", "x": 100, "y": 200, "semantic_label": "Confirm"},
            },
            {"success": True, "changed": True, "reason": "ok", "action": {"type": "type", "text": "YES"}},
            {"success": True, "changed": True, "reason": "ok", "action": {"type": "key", "keys": ["enter"]}},
        ]

        Orchestrator._maybe_synthesize_skill_from_trace(orchestrator, trace, step, "cancel subscription x")

        orchestrator.memory.save_skill.assert_called_once()
        kwargs = orchestrator.memory.save_skill.call_args.kwargs
        self.assertIn("self_healed", kwargs.get("tags", []))
        self.assertEqual(kwargs.get("plan_step_id"), 7)
        self.assertGreaterEqual(len(kwargs.get("actions", [])), 3)

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

    def test_hitl_prompt_allows_execution(self):
        engine = self._make_engine()
        engine.policy_engine.evaluate = MagicMock(
            return_value=PolicyDecision(allowed=True, reason="destructive shell operation", hitl_required=True)
        )
        engine.hid_driver.left_click = MagicMock(return_value=ActionResult(True, "hid"))

        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="y"):
            result = engine.execute({"type": "left_click", "x": 10, "y": 20})

        self.assertTrue(result.success)
        engine.hid_driver.left_click.assert_called_once_with(10, 20)

    def test_hitl_prompt_denies_execution(self):
        engine = self._make_engine()
        engine.policy_engine.evaluate = MagicMock(
            return_value=PolicyDecision(allowed=True, reason="destructive shell operation", hitl_required=True)
        )
        engine.hid_driver.left_click = MagicMock(return_value=ActionResult(True, "hid"))

        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="n"):
            result = engine.execute({"type": "left_click", "x": 10, "y": 20})

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "human confirmation denied")
        engine.hid_driver.left_click.assert_not_called()

    def test_vision_capture_applies_sensitive_redaction(self):
        display = DisplayInfo(
            logical_width=32,
            logical_height=32,
            physical_width=32,
            physical_height=32,
            scale_factor=1.0,
        )
        settings = Settings(enable_sensitive_vision_redaction=True, encode_format="PNG")
        with patch("macos_cua_agent.drivers.vision_pipeline.get_display_info", return_value=display):
            pipeline = VisionPipeline(settings)

        source = Image.new("RGB", (32, 32), color=(255, 255, 255))
        redacted = Image.new("RGB", (32, 32), color=(0, 0, 0))

        with patch.object(pipeline, "_grab_frame", return_value=source), patch(
            "macos_cua_agent.drivers.vision_pipeline.redact_sensitive_regions",
            return_value=(redacted, 1),
        ) as mock_redact:
            image_b64 = pipeline.capture_base64()

        mock_redact.assert_called_once()
        decoded = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 0))

    def test_vision_capture_skips_redaction_when_disabled(self):
        display = DisplayInfo(
            logical_width=32,
            logical_height=32,
            physical_width=32,
            physical_height=32,
            scale_factor=1.0,
        )
        settings = Settings(enable_sensitive_vision_redaction=False, encode_format="PNG")
        with patch("macos_cua_agent.drivers.vision_pipeline.get_display_info", return_value=display):
            pipeline = VisionPipeline(settings)

        source = Image.new("RGB", (32, 32), color=(255, 255, 255))

        with patch.object(pipeline, "_grab_frame", return_value=source), patch(
            "macos_cua_agent.drivers.vision_pipeline.redact_sensitive_regions",
            return_value=(source, 0),
        ) as mock_redact:
            pipeline.capture_base64()

        mock_redact.assert_not_called()

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

    def test_state_manager_records_verification_failure(self):
        state = StateManager()
        self.assertEqual(state.failure_count, 0)

        state.record_verification_failure("no_change", action={"type": "left_click"})

        self.assertEqual(state.failure_count, 1)
        self.assertTrue(any("verification_failure:left_click:no_change" in h for h in state.history))

    def test_state_manager_normalizes_verification_contract(self):
        state = StateManager()
        contract = state.normalize_verification_contract(
            {"sensor": "VISION_FULL", "expected_state": "text_exists:Done", "timeout_seconds": 99},
            fallback_sensor="a11y_tree",
        )
        self.assertEqual(contract.sensor, "vision_full")
        self.assertEqual(contract.expected_state, "text_exists:Done")
        self.assertEqual(contract.timeout_seconds, 30)
        self.assertLess(state.sensor_rank("none"), state.sensor_rank("vision_full"))

    def test_state_manager_verify_after_false_forces_none_contract(self):
        state = StateManager()
        contract = state.normalize_verification_contract(
            {"sensor": "a11y_tree", "expected_state": "text_exists:OK", "timeout_seconds": 10},
            fallback_sensor="a11y_tree",
            verify_after=False,
        )
        self.assertEqual(contract.sensor, "none")
        self.assertEqual(contract.timeout_seconds, 1)

    def test_orchestrator_resolves_contract_with_step_expected_state(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.settings = Settings()
        step = Step(
            id=1,
            description="submit login",
            success_criteria="dashboard appears",
            expected_state="text_exists:Dashboard",
        )
        state = StateManager()
        contract = Orchestrator._resolve_verification_contract(orchestrator, state, {"type": "left_click"}, step)
        self.assertEqual(contract.sensor, "a11y_tree")
        self.assertEqual(contract.expected_state, "text_exists:Dashboard")

    def test_orchestrator_a11y_fallback_accepts_visual_change_when_a11y_unavailable(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._verify_a11y_tree = MagicMock(return_value=(False, "a11y capture failed", None))
        orchestrator._run_visual_verification = MagicMock(
            return_value={
                "passed": True,
                "reason": "visual changed",
                "sensor": "vision_full",
                "changed": True,
                "next_frame": "frame_after",
                "next_hash": "hash_after",
                "hash_distance": 5,
                "ssim_score": 0.8,
                "ax_tree_after": None,
                "ax_changed": False,
                "note": "verification:visual",
                "force_vision_next_turn": True,
            }
        )

        result = Orchestrator._run_verification_contract(
            orchestrator,
            action={"type": "left_click"},
            contract=VerificationContract(sensor="a11y_tree", timeout_seconds=1),
            current_frame="frame_before",
            current_hash="hash_before",
            ax_tree_before=None,
            telemetry_before={},
            global_hotkeys=set(),
            phash_static_threshold=4,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["sensor"], "a11y_tree")
        self.assertEqual(result["note"], "verification:a11y_tree:fallback")
        self.assertIn("visual fallback detected change", result["reason"])

    def test_orchestrator_a11y_fallback_keeps_failure_for_real_a11y_mismatch(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._verify_a11y_tree = MagicMock(return_value=(False, "a11y text not found", {"role": "AXWindow"}))
        orchestrator._run_visual_verification = MagicMock(
            return_value={
                "passed": True,
                "reason": "visual changed",
                "sensor": "vision_full",
                "changed": True,
                "next_frame": "frame_after",
                "next_hash": "hash_after",
                "hash_distance": 5,
                "ssim_score": 0.8,
                "ax_tree_after": {"role": "AXWindow"},
                "ax_changed": True,
                "note": "verification:visual",
                "force_vision_next_turn": True,
            }
        )

        result = Orchestrator._run_verification_contract(
            orchestrator,
            action={"type": "left_click"},
            contract=VerificationContract(sensor="a11y_tree", expected_state="text_exists:Done", timeout_seconds=1),
            current_frame="frame_before",
            current_hash="hash_before",
            ax_tree_before={"role": "AXWindow"},
            telemetry_before={},
            global_hotkeys=set(),
            phash_static_threshold=4,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["sensor"], "a11y_tree")
        self.assertEqual(result["reason"], "a11y text not found")
        self.assertEqual(result["note"], "verification:a11y_tree:timeout")

    def test_os_telemetry_state_change_ignores_timestamp_noise(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        contract = VerificationContract(sensor="os_telemetry", timeout_seconds=2)
        orchestrator._read_clipboard_snapshot = MagicMock(return_value="stable")

        before_snapshot = Orchestrator._collect_os_telemetry_snapshot(orchestrator, contract)
        after_snapshot = Orchestrator._collect_os_telemetry_snapshot(orchestrator, contract)
        changed, reason = Orchestrator._evaluate_os_telemetry_state(
            orchestrator,
            "state_change",
            before_snapshot,
            after_snapshot,
        )

        self.assertNotIn("timestamp", before_snapshot)
        self.assertNotIn("timestamp", after_snapshot)
        self.assertFalse(changed)
        self.assertEqual(reason, "os telemetry unchanged")

    def test_fast_path_without_reflector_requires_change_signal(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.logger = MagicMock()
        orchestrator._persist_fast_path_episode = MagicMock()

        skill = MagicMock()
        skill.id = "skill-1"
        skill.name = "demo-skill"
        skill.actions = [{"type": "left_click", "x": 10, "y": 10}]
        match = MagicMock()
        match.skill = skill
        match.strategy = "keyword"
        match.score = 9.0

        orchestrator.memory = MagicMock()
        orchestrator.memory.select_fast_path_skill.return_value = match

        orchestrator.reflector = MagicMock()
        orchestrator.reflector.available = False

        orchestrator.computer = MagicMock()
        orchestrator.computer.execute.return_value = ActionResult(True, "ok")
        orchestrator.computer.capture_with_hash.return_value = ("frame_after", "hash_after")
        orchestrator.computer.has_changed.return_value = False

        result = Orchestrator._attempt_fast_path(orchestrator, "do the thing", "frame_before", "hash_before")

        self.assertTrue(result["attempted"])
        self.assertFalse(result["success"])
        self.assertIn("fast_path_change_check:unchanged", result["history"])
        orchestrator._persist_fast_path_episode.assert_not_called()

    def test_requires_state_change_detects_avancar_like_actions(self):
        orchestrator = Orchestrator.__new__(Orchestrator)

        step = Step(id=1, description="Click advance to continue", success_criteria="Next page is visible")
        action = {"type": "left_click", "semantic_label": "Avancar"}

        requires_change = Orchestrator._requires_state_change(orchestrator, action, step)
        self.assertTrue(requires_change)

    @patch("macos_cua_agent.drivers.browser_driver.subprocess.run")
    def test_applescript_timeout_is_handled(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["osascript"], timeout=0.1)
        driver = BrowserDriver(self.settings)

        result = driver._run_arg_applescript("Safari", "return \"ok\"", [], "timeout_test", timeout=0.1)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.reason)

if __name__ == '__main__':
    unittest.main()
