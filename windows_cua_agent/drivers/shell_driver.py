from __future__ import annotations

import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.drivers import BaseShellDriver
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


_ABS_DRIVE_RE = re.compile(r"(?i)\b[A-Z]:\\")
_UNC_RE = re.compile(r"^\\\\")


class ShellDriver(BaseShellDriver):
    """Runs sandboxed PowerShell commands inside a constrained workspace."""

    def __init__(self, settings: Settings, allowed_commands: set[str] | None = None) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_shell)
        self.allowed_script_extensions = settings.script_extension_allowlist()

        self.workspace_root = Path(settings.shell_workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Default allowlist is intentionally small; can be expanded via SHELL_ALLOWED_COMMANDS.
        if allowed_commands:
            self.allowed_commands = {cmd.strip().lower() for cmd in allowed_commands if cmd and cmd.strip()}
        else:
            allow_env = (settings.shell_allowed_commands or "").strip()
            if allow_env:
                self.allowed_commands = {cmd.strip().lower() for cmd in allow_env.split(",") if cmd.strip()}
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

    def _execute_script_op(self, action: dict) -> ActionResult:
        operation = str(action.get("operation") or action.get("action") or "").strip().lower()
        if operation not in {"write", "read", "run"}:
            return ActionResult(success=False, reason=f"unknown script operation: {operation or 'none'}")

        cwd = self._resolve_cwd(action.get("cwd"))
        if cwd is None:
            return ActionResult(success=False, reason="cwd outside workspace")

        script_path = self._resolve_script_path(action.get("path"), cwd=cwd)
        if script_path is None:
            return ActionResult(success=False, reason="script path outside workspace")

        if operation == "write":
            return self._script_write(script_path, action)
        if operation == "read":
            return self._script_read(script_path)
        return self._script_run(script_path, action, cwd=cwd)

    def _script_write(self, script_path: Path, action: dict) -> ActionResult:
        ext = script_path.suffix.lower()
        if ext not in self.allowed_script_extensions:
            return ActionResult(success=False, reason=f"script extension not allowed: {ext or '<none>'}")

        content = str(action.get("content") or "")
        max_bytes = int(self.settings.script_max_file_bytes or 131072)
        size = len(content.encode("utf-8", errors="ignore"))
        if size > max_bytes:
            return ActionResult(success=False, reason=f"script content exceeds {max_bytes} bytes")

        overwrite = bool(action.get("overwrite", True))
        if script_path.exists() and not overwrite:
            return ActionResult(success=False, reason="script already exists and overwrite=false")

        if not self.enabled:
            self.logger.info("Shell disabled; dry-run for script write: %s", script_path)
            return ActionResult(success=True, reason="script write dry-run (disabled)")

        try:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            return ActionResult(success=False, reason=f"script write failed: {exc}")

        rel = self._to_workspace_relative(script_path)
        return ActionResult(
            success=True,
            reason=f"script written: {rel}",
            metadata={"operation": "write", "path": rel, "bytes": size},
        )

    def _script_read(self, script_path: Path) -> ActionResult:
        if not script_path.exists():
            return ActionResult(success=False, reason="script file not found")
        if not script_path.is_file():
            return ActionResult(success=False, reason="script path is not a file")

        if not self.enabled:
            self.logger.info("Shell disabled; dry-run for script read: %s", script_path)
            return ActionResult(success=True, reason="script read dry-run (disabled)")

        try:
            raw = script_path.read_bytes()
        except Exception as exc:
            return ActionResult(success=False, reason=f"script read failed: {exc}")

        limit = int(self.settings.shell_max_output_bytes or 65536)
        truncated = len(raw) > limit
        content = raw[:limit].decode("utf-8", errors="replace")
        rel = self._to_workspace_relative(script_path)
        return ActionResult(
            success=True,
            reason=f"script read: {rel}",
            metadata={
                "operation": "read",
                "path": rel,
                "stdout": content,
                "content": content,
                "truncated": truncated,
            },
        )

    def _script_run(self, script_path: Path, action: dict, *, cwd: Path) -> ActionResult:
        if not script_path.exists():
            return ActionResult(success=False, reason="script file not found")
        if not script_path.is_file():
            return ActionResult(success=False, reason="script path is not a file")

        ext = script_path.suffix.lower()
        if ext not in self.allowed_script_extensions:
            return ActionResult(success=False, reason=f"script extension not allowed: {ext or '<none>'}")

        raw_args = action.get("args")
        script_args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
        argv = self._build_script_argv(script_path, script_args)
        if not argv:
            return ActionResult(success=False, reason=f"unsupported runtime for extension: {ext or '<none>'}")

        if not self.enabled:
            self.logger.info("Shell disabled; dry-run for script run: %s", argv)
            return ActionResult(success=True, reason="script run dry-run (disabled)")

        timeout_s = self._script_timeout(action.get("runtime_seconds"))
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                reason="script timeout",
                metadata={"stdout": "", "stderr": "timeout"},
            )
        except Exception as exc:
            return ActionResult(success=False, reason=f"script run failed: {exc}")

        stdout = (completed.stdout or "")[: self.settings.shell_max_output_bytes]
        stderr = (completed.stderr or "")[: self.settings.shell_max_output_bytes]
        rel = self._to_workspace_relative(script_path)
        return ActionResult(
            success=completed.returncode == 0,
            reason=f"script exit {completed.returncode}",
            metadata={
                "operation": "run",
                "path": rel,
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
        normalized = token.replace("/", "\\")
        if _UNC_RE.match(normalized) or _ABS_DRIVE_RE.search(normalized):
            return None
        path = Path(normalized)
        if path.is_absolute():
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

    def _script_timeout(self, runtime_value: object) -> int:
        try:
            requested = int(runtime_value) if runtime_value is not None else int(self.settings.shell_max_runtime_s)
        except Exception:
            requested = int(self.settings.shell_max_runtime_s)
        return max(1, min(requested, int(self.settings.shell_max_runtime_s)))

    def _build_script_argv(self, script_path: Path, args: list[str]) -> list[str]:
        ext = script_path.suffix.lower()
        if ext == ".py":
            return [sys.executable, str(script_path), *args]
        if ext == ".ps1":
            return ["powershell", "-NoProfile", "-NonInteractive", "-File", str(script_path), *args]
        if ext in {".bat", ".cmd"}:
            return ["cmd", "/c", str(script_path), *args]
        if ext == ".js":
            node_bin = shutil.which("node")
            if node_bin:
                return [node_bin, str(script_path), *args]
            cscript_bin = shutil.which("cscript")
            if cscript_bin:
                return [cscript_bin, "//NoLogo", str(script_path), *args]
        return []

    def _to_workspace_relative(self, target: Path) -> str:
        try:
            return str(target.relative_to(self.workspace_root))
        except Exception:
            return str(target)

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
