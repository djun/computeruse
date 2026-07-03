from __future__ import annotations

from importlib.resources import as_file, files

from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.computer.base_computer import DriverBackedComputerAdapter
from cua_agent.computer.drivers import BaseVisionPipeline
from cua_agent.utils.config import Settings

from windows_cua_agent.drivers.action_engine import ActionEngine
from windows_cua_agent.drivers.vision_pipeline import VisionPipeline
from windows_cua_agent.policies.windows_policy_engine import WindowsPolicyEngine
from windows_cua_agent.utils.health import run_permission_health_checks
from windows_cua_agent.utils.windows_integration import ensure_dpi_awareness, get_display_info, get_system_info


class WindowsComputer(DriverBackedComputerAdapter):
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

        self.vision: BaseVisionPipeline = VisionPipeline(settings)
        self.action_engine = ActionEngine(settings, policy_engine, vision_pipeline=self.vision)

    def run_health_checks(self, settings: Settings, logger=None) -> None:
        run_permission_health_checks(settings, logger=logger)


def create_computer(settings: Settings) -> ComputerAdapter:
    return WindowsComputer(settings)
