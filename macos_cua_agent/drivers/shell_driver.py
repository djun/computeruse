from __future__ import annotations

import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.shell_driver_base import SharedShellDriver
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class ShellDriver(SharedShellDriver):
    """Runs sandboxed shell commands inside a constrained workspace."""
    RUNNABLE_SCRIPT_EXTENSIONS = {".py", ".sh", ".js", ".ps1"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_shell)
        configured_extensions = settings.script_extension_allowlist()
        unsupported = configured_extensions.difference(self.RUNNABLE_SCRIPT_EXTENSIONS)
        self.allowed_script_extensions = configured_extensions.intersection(self.RUNNABLE_SCRIPT_EXTENSIONS)
        if unsupported:
            self.logger.info(
                "Ignoring unsupported script extensions on macOS shell driver: %s",
                sorted(unsupported),
            )

        self.workspace_root = Path(settings.shell_workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def execute(self, action: dict) -> ActionResult:
        if not self.settings.allows_shell_actions():
            return ActionResult(
                success=False,
                reason=f"execution profile '{self.settings.execution_profile}' blocks shell actions",
            )

        action_type = str(action.get("type") or "").strip().lower()
        if action_type == "script_op":
            return self._execute_script_op(action)

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

    def _resolve_script_path(self, raw_path: object, *, cwd: Path) -> Optional[Path]:
        token = str(raw_path or "").strip()
        if not token:
            return None
        normalized = token.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or normalized.startswith("~"):
            return None
        try:
            candidate = (cwd / path).resolve()
        except Exception:
            return None
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError:
            self.logger.warning("Blocked script path escape: %s", candidate)
            return None
        return candidate

    def _build_script_argv(self, script_path: Path, args: list[str]) -> list[str]:
        ext = script_path.suffix.lower()
        if ext == ".py":
            return [sys.executable, str(script_path), *args]
        if ext == ".sh":
            return ["/bin/bash", str(script_path), *args]
        if ext == ".js":
            node_bin = shutil.which("node")
            if not node_bin:
                return []
            return [node_bin, str(script_path), *args]
        if ext == ".ps1":
            ps_bin = shutil.which("pwsh") or shutil.which("powershell")
            if not ps_bin:
                return []
            return [ps_bin, "-NoProfile", "-NonInteractive", "-File", str(script_path), *args]
        return []

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
