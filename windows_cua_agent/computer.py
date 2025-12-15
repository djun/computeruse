from __future__ import annotations

from importlib.resources import as_file, files

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.utils.config import Settings

from windows_cua_agent.drivers.action_engine import ActionEngine
from windows_cua_agent.drivers.vision_pipeline import VisionPipeline
from windows_cua_agent.policies.windows_policy_engine import WindowsPolicyEngine
from windows_cua_agent.utils.health import run_permission_health_checks
from windows_cua_agent.utils.windows_integration import ensure_dpi_awareness, get_display_info, get_system_info


class WindowsComputer(ComputerAdapter):
    platform_name = "Windows"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        ensure_dpi_awareness(logger_name=__name__)
        self.display = get_display_info()
        self.system_info = get_system_info()

        self.global_hotkeys = {
            ("alt", "tab"),
            ("ctrl", "esc"),
            ("win",),
        }

        rules_resource = files("windows_cua_agent.policies").joinpath("windows_safety_rules.yaml")
        with as_file(rules_resource) as rules_path:
            policy_engine = WindowsPolicyEngine(str(rules_path), settings)

        self.vision = VisionPipeline(settings)
        self.action_engine = ActionEngine(settings, policy_engine)

    def run_health_checks(self, settings: Settings, logger=None) -> None:
        run_permission_health_checks(settings, logger=logger)

    def capture_base64(self) -> str:
        return self.vision.capture_base64()

    def capture_with_hash(self) -> tuple[str, str]:
        return self.vision.capture_with_hash()

    def hash_base64(self, image_b64: str) -> str:
        return self.vision.hash_base64(image_b64)

    def hash_distance(self, hash_a: str | None, hash_b: str | None) -> int:
        return self.vision.hash_distance(hash_a, hash_b)

    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool:
        return self.vision.has_changed(previous_b64, current_b64, threshold=threshold)

    def structural_similarity(self, previous_b64: str, current_b64: str) -> float | None:
        return self.vision.structural_similarity(previous_b64, current_b64)

    def detect_ui_elements(self, image_b64: str) -> list[dict]:
        return self.vision.detect_ui_elements(image_b64)

    def get_active_window_tree(self, max_depth: int = 5):
        return self.action_engine.accessibility_driver.get_active_window_tree(max_depth=max_depth)

    def execute(self, action: dict):
        return self.action_engine.execute(action)


def create_computer(settings: Settings) -> ComputerAdapter:
    return WindowsComputer(settings)
