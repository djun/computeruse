from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class ShellDriver:
    """Runs sandboxed shell commands inside a constrained workspace."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_shell)

        self.workspace_root = Path(settings.shell_workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def execute(self, action: dict) -> ActionResult:
        cmd_raw = action.get("cmd") or action.get("command")
        if not cmd_raw:
            return ActionResult(success=False, reason="no command provided")

        if isinstance(cmd_raw, str):
            argv = shlex.split(cmd_raw)
        else:
            argv = list(cmd_raw)

        if not argv:
            return ActionResult(success=False, reason="empty command")

        cwd = self._resolve_cwd(action.get("cwd"))
        if cwd is None:
            return ActionResult(success=False, reason="cwd outside workspace")

        if not self.enabled:
            self.logger.info("Shell disabled; dry-run for command: %s (cwd=%s)", argv, cwd)
            return ActionResult(success=True, reason="shell dry-run (disabled)")

        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.settings.shell_max_runtime_s,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                reason="shell timeout",
                metadata={"stdout": "", "stderr": "timeout"},
            )
        except Exception as exc:
            self.logger.error("Shell execution failed: %s", exc)
            return ActionResult(success=False, reason=str(exc))

        stdout = (completed.stdout or "")[: self.settings.shell_max_output_bytes]
        stderr = (completed.stderr or "")[: self.settings.shell_max_output_bytes]
        success = completed.returncode == 0

        return ActionResult(
            success=success,
            reason=f"exit {completed.returncode}",
            metadata={
                "argv": argv,
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def _resolve_cwd(self, cwd: Optional[str]) -> Optional[Path]:
        base = self.workspace_root
        target = (base / cwd).resolve() if cwd else base
        try:
            target.relative_to(base)
        except ValueError:
            self.logger.warning("Blocked cwd escape: %s", target)
            return None
        target.mkdir(parents=True, exist_ok=True)
        return target
