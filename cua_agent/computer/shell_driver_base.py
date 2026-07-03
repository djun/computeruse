"""Shared sandboxed script-op handling for OS shell drivers.

Subclasses build the workspace and allowlists in ``__init__`` (setting
``self.settings``, ``self.logger``, ``self.enabled``,
``self.allowed_script_extensions`` and ``self.workspace_root``), implement
``execute`` for raw shell commands (the execution model differs per OS: direct
argv on macOS, a validated PowerShell wrapper on Windows), and provide the
OS-specific path/interpreter hooks below. The ``script_op`` write/read/run flow
is identical across platforms and lives here.
"""

from __future__ import annotations

import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from cua_agent.agent.state_manager import ActionResult
from cua_agent.computer.drivers import BaseShellDriver


class SharedShellDriver(BaseShellDriver):
    """Runs sandboxed script operations inside a constrained workspace."""

    @abstractmethod
    def _resolve_cwd(self, cwd: Optional[str]) -> Optional[Path]:
        raise NotImplementedError

    @abstractmethod
    def _resolve_script_path(self, raw_path: object, *, cwd: Path) -> Optional[Path]:
        raise NotImplementedError

    @abstractmethod
    def _build_script_argv(self, script_path: Path, args: list[str]) -> list[str]:
        raise NotImplementedError

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

    def _script_timeout(self, runtime_value: object) -> int:
        try:
            requested = int(runtime_value) if runtime_value is not None else int(self.settings.shell_max_runtime_s)
        except Exception:
            requested = int(self.settings.shell_max_runtime_s)
        return max(1, min(requested, int(self.settings.shell_max_runtime_s)))

    def _to_workspace_relative(self, target: Path) -> str:
        try:
            return str(target.relative_to(self.workspace_root))
        except Exception:
            return str(target)
