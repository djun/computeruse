from __future__ import annotations

from importlib.resources import as_file, files

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.computer.base_computer import DriverBackedComputerAdapter
from cua_agent.computer.drivers import BaseVisionPipeline
from cua_agent.policies.policy_engine import PolicyEngine
from cua_agent.utils.config import Settings
from macos_cua_agent.drivers.action_engine import ActionEngine
from macos_cua_agent.drivers.vision_pipeline import VisionPipeline
from macos_cua_agent.utils.health import run_permission_health_checks
from macos_cua_agent.utils.macos_integration import get_display_info, get_system_info


class MacOSComputer(DriverBackedComputerAdapter):
    platform_name = "macOS"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.display = get_display_info()
        self.system_info = get_system_info()
        self.global_hotkeys = {
            ("cmd", "space"),
            ("command", "space"),
            ("cmd", "tab"),
            ("command", "tab"),
        }

        rules_resource = files("cua_agent.policies").joinpath("safety_rules.yaml")
        with as_file(rules_resource) as rules_path:
            policy_engine = PolicyEngine(str(rules_path), settings)

        self.vision: BaseVisionPipeline = VisionPipeline(settings)
        self.action_engine = ActionEngine(settings, policy_engine, vision_pipeline=self.vision)

    def run_health_checks(self, settings: Settings, logger=None) -> None:
        run_permission_health_checks(settings, logger=logger)


def create_computer(settings: Settings) -> ComputerAdapter:
    return MacOSComputer(settings)
