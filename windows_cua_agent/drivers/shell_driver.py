from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


_ABS_DRIVE_RE = re.compile(r"(?i)\b[A-Z]:\\")
_UNC_RE = re.compile(r"^\\\\")


class ShellDriver:
    """Runs sandboxed PowerShell commands inside a constrained workspace."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_shell)

        self.workspace_root = Path(settings.shell_workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Default allowlist is intentionally small; can be expanded via SHELL_ALLOWED_COMMANDS.
        allow_env = (settings.shell_allowed_commands or "").strip()
        if allow_env:
            self.allowed_commands = {cmd.strip() for cmd in allow_env.split(",") if cmd.strip()}
        else:
            self.allowed_commands = {
                "dir",
                "type",
                "copy",
                "move",
                "del",
                "select-string",
                "get-childitem",
                "get-content",
            }

    def execute(self, action: dict) -> ActionResult:
        cmd_raw = action.get("cmd") or action.get("command")
        if not cmd_raw:
            return ActionResult(success=False, reason="no command provided")
        if not isinstance(cmd_raw, str):
            # Windows adapter expects PowerShell commands as a string.
            cmd_raw = " ".join(str(x) for x in cmd_raw)

        cwd = self._resolve_cwd(action.get("cwd"))
        if cwd is None:
            return ActionResult(success=False, reason="cwd outside workspace")

        # Fast-path: policy-style checks (best effort; driver still enforces workspace via cwd).
        ok, reason = self._validate_command(cmd_raw, cwd=cwd)
        if not ok:
            return ActionResult(success=False, reason=reason)

        if not self.enabled:
            self.logger.info("Shell disabled; dry-run for command: %s (cwd=%s)", cmd_raw, cwd)
            return ActionResult(success=True, reason="shell dry-run (disabled)")

        wrapper = (
            "$ErrorActionPreference = 'Stop';"
            "try {"
            f"{cmd_raw};"
            "exit 0"
            "} catch {"
            "Write-Error $_;"
            "exit 1"
            "}"
        )

        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", wrapper],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.settings.shell_max_runtime_s,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(success=False, reason="shell timeout", metadata={"stdout": "", "stderr": "timeout"})
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
                "argv": ["powershell", "-Command", cmd_raw],
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def _resolve_cwd(self, cwd: Optional[str]) -> Optional[Path]:
        base = self.workspace_root
        if cwd:
            # Normalize separators; treat absolute paths as-is but still enforce sandbox.
            normalized = cwd.replace("/", "\\")
            target = Path(normalized)
            if not target.is_absolute():
                target = base / target
        else:
            target = base

        try:
            target = target.resolve()
        except Exception:
            target = base

        try:
            target.relative_to(base)
        except ValueError:
            self.logger.warning("Blocked cwd escape: %s", target)
            return None

        target.mkdir(parents=True, exist_ok=True)
        return target

    def _validate_command(self, cmd: str, *, cwd: Path) -> tuple[bool, str]:
        """
        Best-effort safety validation.

        - Only allow commands whose first token(s) are allowlisted (supports simple pipelines).
        - Block obvious escape hatches (redirection, background operators, UNC paths).
        - Block absolute paths outside the workspace root.
        """
        stripped = (cmd or "").strip()
        if not stripped:
            return False, "empty command"

        # Disallow common PowerShell chaining/redirection operators.
        forbidden = ["&&", "||", ">", "<", "`n", "`r"]
        if any(op in stripped for op in forbidden):
            return False, "shell operator not allowed"

        # Split by pipeline and validate each stage command.
        stages = [s.strip() for s in stripped.split("|") if s.strip()]
        for stage in stages:
            try:
                tokens = shlex.split(stage, posix=False)
            except Exception:
                tokens = stage.split()
            if not tokens:
                return False, "empty pipeline stage"
            cmd_name = (tokens[0] or "").strip().lower()
            if cmd_name not in self.allowed_commands:
                return False, f"command not allowed: {tokens[0]}"

            # Basic path escape detection in args.
            for tok in tokens[1:]:
                t = tok.strip().strip("'\"")
                if not t:
                    continue
                if ".." in t.replace("/", "\\"):
                    return False, "path traversal not allowed"
                if _UNC_RE.match(t):
                    return False, "UNC paths not allowed"
                if _ABS_DRIVE_RE.search(t):
                    if not self._path_within_workspace(t, cwd=cwd):
                        return False, "absolute path outside workspace"

        return True, ""

    def _path_within_workspace(self, path_str: str, *, cwd: Path) -> bool:
        try:
            p = Path(path_str)
            if not p.is_absolute():
                p = cwd / p
            resolved = p.resolve(strict=False)
            resolved.relative_to(self.workspace_root)
            return True
        except Exception:
            return False

