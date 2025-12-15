from __future__ import annotations

import re
from typing import Any, Dict

from cua_agent.policies.policy_engine import PolicyDecision, PolicyEngine
from cua_agent.utils.config import Settings


class WindowsPolicyEngine(PolicyEngine):
    """
    Windows-specific policy evaluation.

    The core PolicyEngine's shell allowlist is Unix-path based; on Windows we instead
    treat `sandbox_shell` commands as PowerShell/cmdlet strings and validate them
    against a basename allowlist.
    """

    def __init__(self, rules_path: str, settings: Settings | None = None) -> None:
        super().__init__(rules_path, settings=settings)
        self._settings = settings

    def evaluate(self, action: Dict[str, Any]) -> PolicyDecision:
        action_type = action.get("type") or action.get("action")
        if action_type == "sandbox_shell":
            return self._evaluate_windows_shell(action)
        return super().evaluate(action)

    def _evaluate_windows_shell(self, action: Dict[str, Any]) -> PolicyDecision:
        cmd_raw = action.get("cmd") or action.get("command") or ""
        cmd = cmd_raw if isinstance(cmd_raw, str) else " ".join(str(x) for x in cmd_raw)
        cmd_stripped = cmd.strip()
        if not cmd_stripped:
            return PolicyDecision(False, "empty command")

        # Allowlist from env/settings takes priority; fall back to rules file.
        allow_env = (getattr(self._settings, "shell_allowed_commands", "") or "").strip() if self._settings else ""
        if allow_env:
            allow = {c.strip().lower() for c in allow_env.split(",") if c.strip()}
        else:
            allow = {c.lower() for c in self.rules.get("allowed_shell_basenames", []) if isinstance(c, str)}

        # Heuristic: validate each pipeline stage's command token.
        stages = [s.strip() for s in cmd_stripped.split("|") if s.strip()]
        for stage in stages:
            token = stage.split(None, 1)[0].strip().strip("\"'").lower()
            if not token:
                return PolicyDecision(False, "empty pipeline stage")
            if allow and token not in allow:
                return PolicyDecision(False, f"command not allowed: {token}")

        # HITL triggers for destructive operations and script execution.
        lowered = cmd_stripped.lower()
        destructive = [
            "rd /s",
            "rmdir /s",
            "del /s",
            "remove-item",
            "rm -rf",
            "format ",
        ]
        if any(pat in lowered for pat in destructive):
            return PolicyDecision(True, "destructive shell operation", hitl_required=True)

        if re.search(r"(?i)\b(powershell|pwsh)\b.*-file\b", cmd_stripped):
            return PolicyDecision(True, "script execution", hitl_required=True)
        if re.search(r"(?i)\.(ps1|bat|vbs)\b", cmd_stripped):
            return PolicyDecision(True, "script execution", hitl_required=True)

        return PolicyDecision(True)

