from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger

DEFAULT_RULES = {
    "blocked_actions": ["shell_command"],
    "blocked_bundle_ids": ["com.apple.keychainaccess"],
    "hitl_actions": ["erase_disk", "format_disk"],
    "allowed_shell_basenames": [], # Deprecated but kept for compatibility
    "blocked_shell_basenames": [], # Deprecated
    "exclusion_zones": [], # List of {x, y, w, h, label}
}


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    hitl_required: bool = False


class PolicyEngine:
    """Evaluates proposed actions against configured safety rules."""
    
    # Secure Allowlist: Absolute Path -> Allowed Args (or "*" for all)
    ALLOWED_COMMANDS = {
        "/bin/ls": ["*"],
        "/bin/echo": ["*"],
        "/usr/bin/grep": ["*"],
        "/usr/bin/wc": ["*"],
        "/usr/bin/git": ["status", "log", "diff", "show", "checkout", "branch"],
    }

    def __init__(self, rules_path: str, settings: Settings | None = None) -> None:
        self.logger = get_logger(__name__)
        self.rules = self._load_rules(rules_path)
        self._apply_overrides_from_settings(settings)

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        if not os.path.exists(rules_path):
            self.logger.info("safety_rules.yaml missing; using defaults.")
            return dict(DEFAULT_RULES)
        try:
            import yaml  # type: ignore
        except Exception as exc:
            self.logger.warning("PyYAML unavailable (%s); using defaults.", exc)
            return dict(DEFAULT_RULES)

        with open(rules_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        merged = dict(DEFAULT_RULES)
        merged.update(loaded)
        return merged

    def _apply_overrides_from_settings(self, settings: Settings | None) -> None:
        if not settings:
            return

        allow_env = settings.shell_allowed_commands
        if allow_env:
            allowlist: List[str] = [cmd.strip() for cmd in allow_env.split(",") if cmd.strip()]
            if allowlist:
                self.rules["allowed_shell_basenames"] = allowlist
                self.logger.info("Applied shell allowlist from env: %s", allowlist)

    def evaluate(self, action: Dict[str, Any]) -> PolicyDecision:
        action_type = action.get("type") or action.get("action")
        bundle_id = action.get("bundle_id") or action.get("app")

        if action_type in self.rules.get("blocked_actions", []):
            return PolicyDecision(allowed=False, reason=f"action blocked: {action_type}")
            
        # Spatial Exclusion Check
        x = action.get("x")
        y = action.get("y")
        tx = action.get("target_x")
        ty = action.get("target_y")
        
        zones = self.rules.get("exclusion_zones", [])
        for zone in zones:
            zx, zy, zw, zh = zone.get("x", 0), zone.get("y", 0), zone.get("w", 0), zone.get("h", 0)
            label = zone.get("label", "restricted area")
            
            # Check source point
            if x is not None and y is not None:
                if zx <= x <= zx + zw and zy <= y <= zy + zh:
                    return PolicyDecision(False, f"interaction in exclusion zone: {label}")
            
            # Check target point (drag)
            if tx is not None and ty is not None:
                if zx <= tx <= zx + zw and zy <= ty <= zy + zh:
                    return PolicyDecision(False, f"interaction target in exclusion zone: {label}")

        if action_type == "sandbox_shell":
            cmd_raw = action.get("cmd") or action.get("command") or ""
            argv = []
            if isinstance(cmd_raw, str):
                try:
                    argv = shlex.split(cmd_raw)
                except ValueError:
                    return PolicyDecision(False, "malformed command string")
            elif isinstance(cmd_raw, (list, tuple)):
                argv = list(cmd_raw)
            
            if not argv:
                return PolicyDecision(False, "empty command")

            # 1. Resolve Executable Path
            cmd_name = argv[0]
            resolved_path = shutil.which(cmd_name)
            
            if not resolved_path:
                return PolicyDecision(False, f"command not found: {cmd_name}")

            # 2. Strict Allowlist Check
            if resolved_path not in self.ALLOWED_COMMANDS:
                return PolicyDecision(False, f"command path not allowlisted: {resolved_path}")

            # 3. Argument Validation (Basic)
            allowed_args = self.ALLOWED_COMMANDS.get(resolved_path, [])
            if "*" not in allowed_args:
                # If first arg looks like a subcommand, check it
                if len(argv) > 1:
                    subcommand = argv[1]
                    if not subcommand.startswith("-") and subcommand not in allowed_args:
                         return PolicyDecision(False, f"subcommand not allowed: {subcommand}")
        
        if bundle_id and bundle_id in self.rules.get("blocked_bundle_ids", []):
            return PolicyDecision(allowed=False, reason=f"bundle blocked: {bundle_id}")
        if action_type in self.rules.get("hitl_actions", []):
            return PolicyDecision(allowed=True, hitl_required=True, reason="human confirmation required")
        return PolicyDecision(allowed=True)