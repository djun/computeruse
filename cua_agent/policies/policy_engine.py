"""Policy engine for evaluating actions against safety rules."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List

from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger

DEFAULT_RULES = {
    # Keep both tokens for backwards compatibility across old/new configs.
    "blocked_actions": ["sandbox_shell", "shell_command"],
    "blocked_bundle_ids": ["com.apple.keychainaccess"],
    "hitl_actions": ["erase_disk", "format_disk", "run_javascript"],
    "sensitive_domains": [],
    "allowed_shell_basenames": [],
    "blocked_shell_basenames": [],
    "allowed_script_extensions": [".py", ".sh", ".js", ".ps1", ".bat", ".cmd"],
    "script_hitl_patterns": [
        "import socket",
        "from socket import",
        "import subprocess",
        "from subprocess import",
        "import os",
        "from os import",
        "eval(",
        "exec(",
        "__import__(",
        "powershell -",
        "invoke-webrequest",
        "invoke-restmethod",
    ],
    "exclusion_zones": [],  # List of {x, y, w, h, label}
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
    # Permit basename-based allowlist only when executables resolve to trusted
    # system locations (guards against PATH shadowing inside writable dirs).
    TRUSTED_BIN_DIRS = (
        "/bin",
        "/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/opt/local/bin",
    )

    def __init__(self, rules_path: str, settings: Settings | None = None) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._workspace_root = self._resolve_workspace_root(settings)
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
            allowlist: List[str] = [cmd.strip().lower() for cmd in allow_env.split(",") if cmd.strip()]
            if allowlist:
                self.rules["allowed_shell_basenames"] = allowlist
                self.logger.info("Applied shell allowlist from env: %s", allowlist)

        script_ext_env = str(settings.script_allowed_extensions or "").strip()
        if script_ext_env and self._should_override_script_extensions(script_ext_env):
            self.rules["allowed_script_extensions"] = [
                ext.strip() for ext in script_ext_env.split(",") if ext.strip()
            ]

    def evaluate(self, action: Dict[str, Any]) -> PolicyDecision:
        action_type = action.get("type") or action.get("action")
        bundle_id = action.get("bundle_id") or action.get("app")
        command = action.get("command")
        code_payload = action.get("value") or ""
        page_url = action.get("page_url") or action.get("url") or ""

        blocked_actions = self.rules.get("blocked_actions", [])
        if self._is_action_blocked(action_type, blocked_actions):
            return PolicyDecision(allowed=False, reason=f"action blocked: {action_type}")
        if command and self._is_action_blocked(command, blocked_actions):
            return PolicyDecision(allowed=False, reason=f"command blocked: {command}")

        # Browser safety: block JS on sensitive domains and flag risky payloads
        if action_type == "browser_op" and command == "run_javascript":
            host = self._extract_hostname(page_url)
            for domain in self.rules.get("sensitive_domains", []):
                if host == domain or (domain and host.endswith(f".{domain}")):
                    return PolicyDecision(
                        allowed=False,
                        reason=f"run_javascript blocked on sensitive domain: {host or 'unknown'}",
                    )

            risky = self._contains_dangerous_js(code_payload)
            if risky:
                return PolicyDecision(
                    allowed=True,
                    hitl_required=True,
                    reason=f"run_javascript requires confirmation (risky pattern: {risky})",
                )
            
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

            resolved_realpath = os.path.realpath(resolved_path)
            cmd_basename = os.path.basename(resolved_path).strip().lower()
            if not cmd_basename:
                cmd_basename = os.path.basename(resolved_realpath).strip().lower()

            blocked_basenames = self._normalized_rule_list("blocked_shell_basenames")
            if cmd_basename in blocked_basenames:
                return PolicyDecision(False, f"command blocked: {cmd_basename}")

            allowed_by_path = (
                resolved_path in self.ALLOWED_COMMANDS or resolved_realpath in self.ALLOWED_COMMANDS
            )
            allowed_basenames = self._normalized_rule_list("allowed_shell_basenames")
            allowed_by_basename = (
                cmd_basename in allowed_basenames
                and (
                    self._is_trusted_executable_path(resolved_path)
                    or self._is_trusted_executable_path(resolved_realpath)
                )
            )

            # 2. Allowlist Check (strict path list OR trusted basename list from settings/rules)
            if not allowed_by_path and not allowed_by_basename:
                return PolicyDecision(False, f"command not allowlisted: {cmd_basename or cmd_name}")

            # 3. Argument Validation (Basic)
            if allowed_by_path:
                allowed_args = self.ALLOWED_COMMANDS.get(
                    resolved_path,
                    self.ALLOWED_COMMANDS.get(resolved_realpath, []),
                )
                # If first arg looks like a subcommand, check it
                if "*" not in allowed_args and len(argv) > 1:
                    subcommand = argv[1]
                    if not subcommand.startswith("-") and subcommand not in allowed_args:
                        return PolicyDecision(False, f"subcommand not allowed: {subcommand}")

        if action_type == "script_op":
            decision = self._evaluate_script_action(action)
            if not decision.allowed or decision.hitl_required or decision.reason:
                return decision

        if bundle_id and bundle_id in self.rules.get("blocked_bundle_ids", []):
            return PolicyDecision(allowed=False, reason=f"bundle blocked: {bundle_id}")
        if action_type in self.rules.get("hitl_actions", []):
            return PolicyDecision(allowed=True, hitl_required=True, reason="human confirmation required")
        if command and command in self.rules.get("hitl_actions", []):
            return PolicyDecision(allowed=True, hitl_required=True, reason="human confirmation required")
        return PolicyDecision(allowed=True)

    def _evaluate_script_action(self, action: Dict[str, Any]) -> PolicyDecision:
        operation = str(action.get("operation") or action.get("action") or "").strip().lower()
        if operation not in {"write", "read", "run"}:
            return PolicyDecision(False, f"unknown script operation: {operation or 'none'}")

        raw_path = str(action.get("path") or "").strip()
        if not raw_path:
            return PolicyDecision(False, "script path missing")
        if not self._is_workspace_relative_path(raw_path):
            return PolicyDecision(False, "script path must be relative to workspace")

        resolved_cwd = self._resolve_workspace_cwd(action.get("cwd"))
        if resolved_cwd is None:
            return PolicyDecision(False, "script cwd outside workspace")

        resolved_path = self._resolve_workspace_path(raw_path, base_dir=resolved_cwd)
        if resolved_path is None:
            return PolicyDecision(False, "script path outside workspace")

        allowed_extensions = self._normalized_extensions(self.rules.get("allowed_script_extensions", []))
        ext = resolved_path.suffix.lower()
        if operation in {"write", "run"} and allowed_extensions and ext not in allowed_extensions:
            return PolicyDecision(False, f"script extension not allowed: {ext or '<none>'}")

        if operation == "write":
            content = str(action.get("content") or "")
            max_bytes = int(getattr(self._settings, "script_max_file_bytes", 131072) or 131072)
            if len(content.encode("utf-8", errors="ignore")) > max_bytes:
                return PolicyDecision(False, f"script content exceeds size limit ({max_bytes} bytes)")

            risky_pattern = self._contains_risky_script_content(content)
            if risky_pattern:
                return PolicyDecision(
                    allowed=True,
                    hitl_required=True,
                    reason=f"script write requires confirmation (risky pattern: {risky_pattern})",
                )
            return PolicyDecision(True)

        if operation == "run":
            file_content = self._read_workspace_file_for_policy(resolved_path)
            blocked_cmd = self._contains_blocked_shell_command_in_script(file_content)
            if blocked_cmd:
                return PolicyDecision(
                    allowed=False,
                    reason=f"script run blocked (blocked command in script: {blocked_cmd})",
                )

            dangerous_shell_pattern = self._contains_dangerous_script_shell_pattern(file_content)
            if dangerous_shell_pattern:
                return PolicyDecision(
                    allowed=True,
                    hitl_required=True,
                    reason=(
                        "script run requires confirmation "
                        f"(dangerous shell pattern: {dangerous_shell_pattern})"
                    ),
                )

            risky_pattern = self._contains_risky_script_content(file_content)
            if risky_pattern:
                return PolicyDecision(
                    allowed=True,
                    hitl_required=True,
                    reason=f"script run requires confirmation (risky pattern: {risky_pattern})",
                )
            return PolicyDecision(True)

        return PolicyDecision(True)

    def _is_action_blocked(self, token: Any, blocked_actions: Any) -> bool:
        if not token:
            return False

        if isinstance(blocked_actions, str):
            candidates = [blocked_actions]
        elif isinstance(blocked_actions, (list, tuple, set)):
            candidates = list(blocked_actions)
        else:
            candidates = []

        blocked = {str(item).strip().lower() for item in candidates if str(item).strip()}
        aliases = self._action_aliases(str(token).strip().lower())
        return not blocked.isdisjoint(aliases)

    def _action_aliases(self, action_name: str) -> set[str]:
        aliases = {action_name}
        if action_name == "shell_command":
            aliases.add("sandbox_shell")
        elif action_name == "sandbox_shell":
            aliases.add("shell_command")
        return aliases

    def _normalized_rule_list(self, key: str) -> set[str]:
        raw = self.rules.get(key, [])
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = []
        return {str(item).strip().lower() for item in values if str(item).strip()}

    def _is_trusted_executable_path(self, resolved_path: str) -> bool:
        normalized = os.path.abspath(str(resolved_path or "")).strip()
        if not normalized:
            return False
        return any(
            normalized == prefix or normalized.startswith(f"{prefix}/")
            for prefix in self.TRUSTED_BIN_DIRS
        )

    def _resolve_workspace_root(self, settings: Settings | None) -> Path:
        root = ".agent_shell"
        if settings:
            root = str(getattr(settings, "shell_workspace_root", root) or root)
        return Path(root).expanduser().resolve()

    def _should_override_script_extensions(self, csv_value: str) -> bool:
        requested = self._normalized_extensions(csv_value.split(","))
        if not requested:
            return False
        defaults = self._normalized_extensions(DEFAULT_RULES.get("allowed_script_extensions", []))
        return requested != defaults

    def _resolve_workspace_cwd(self, raw_cwd: Any) -> Path | None:
        base = self._workspace_root
        token = str(raw_cwd or "").strip()
        if not token:
            return base

        normalized = token.replace("\\", "/")
        target = Path(normalized)
        if not target.is_absolute():
            target = base / normalized

        try:
            resolved = target.resolve()
            resolved.relative_to(base)
        except Exception:
            return None
        return resolved

    def _is_workspace_relative_path(self, raw_path: str) -> bool:
        token = str(raw_path or "").strip()
        if not token:
            return False
        normalized = token.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("~") or normalized.startswith("//"):
            return False
        if re.match(r"(?i)^[a-z]:/", normalized):
            return False
        segments = [segment for segment in normalized.split("/") if segment not in {"", "."}]
        if not segments:
            return False
        return ".." not in segments

    def _resolve_workspace_path(self, raw_path: str, *, base_dir: Path | None = None) -> Path | None:
        if not self._is_workspace_relative_path(raw_path):
            return None
        base = base_dir or self._workspace_root
        try:
            base = base.resolve()
            base.relative_to(self._workspace_root)
        except Exception:
            return None
        normalized = str(raw_path).replace("\\", "/")
        candidate = (base / normalized).resolve()
        try:
            candidate.relative_to(self._workspace_root)
        except ValueError:
            return None
        return candidate

    def _normalized_extensions(self, raw: Any) -> set[str]:
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = []

        extensions = set()
        for value in values:
            token = str(value or "").strip().lower()
            if not token:
                continue
            token = "." + token.lstrip(".")
            extensions.add(token)
        return extensions

    def _contains_risky_script_content(self, content: str) -> str:
        payload = str(content or "")
        if not payload:
            return ""
        lower = payload.lower()
        patterns = self.rules.get("script_hitl_patterns", []) or []
        if isinstance(patterns, str):
            values = [patterns]
        elif isinstance(patterns, (list, tuple, set)):
            values = list(patterns)
        else:
            values = []
        for pattern in values:
            token = str(pattern or "").strip().lower()
            if token and token in lower:
                return token
        return ""

    def _contains_blocked_shell_command_in_script(self, content: str) -> str:
        payload = str(content or "")
        if not payload:
            return ""

        blocked = self._normalized_rule_list("blocked_shell_basenames")
        if not blocked:
            return ""

        segments = re.split(r"[;\r\n|&]+", payload)
        for segment in segments:
            token = self._extract_script_command_token(segment)
            if token and token in blocked:
                return token
        return ""

    def _contains_dangerous_script_shell_pattern(self, content: str) -> str:
        payload = str(content or "").lower()
        if not payload:
            return ""

        patterns = [
            "rm -rf",
            "rm -fr",
            "del /s",
            "rd /s",
            "rmdir /s",
            "remove-item -recurse",
            "remove-item -force",
            "format ",
            "diskutil erase",
            "mkfs",
            "dd if=",
            "curl ",
            "wget ",
        ]
        for pattern in patterns:
            if pattern in payload:
                return pattern
        return ""

    def _extract_script_command_token(self, segment: str) -> str:
        lower = str(segment or "").strip().lower()
        if not lower:
            return ""

        # Skip obvious comments and trim common wrappers/prefixes.
        if lower.startswith(("#", "::", "rem ")):
            return ""
        lower = re.sub(r"^\s*@", "", lower)
        lower = re.sub(r"^\s*(?:sudo|command|builtin|exec|call)\s+", "", lower)

        match = re.match(r"([a-z0-9._/-]+)", lower)
        if not match:
            return ""

        token = match.group(1).strip()
        token = token.split("/")[-1]
        token = re.sub(r"\.(exe|bat|cmd|ps1)$", "", token)
        return token

    def _read_workspace_file_for_policy(self, path: Path) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            limit = min(
                int(getattr(self._settings, "script_max_file_bytes", 131072) or 131072),
                262144,
            )
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                return fh.read(limit)
        except Exception:
            return ""

    def _extract_hostname(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.hostname or ""

    def _contains_dangerous_js(self, code: str) -> str:
        """
        Lightweight keyword scan to surface risky JS usage for HitL.
        Returns the matched keyword when found, else empty string.
        """
        if not code:
            return ""
        lower = code.lower()
        keywords = [
            "fetch(",
            "xmlhttprequest",
            "ws://",
            "wss://",
            "document.cookie",
            "localstorage",
            "sessionstorage",
            "indexeddb",
            "eval(",
            "Function(",
        ]
        for kw in keywords:
            if kw in lower:
                return kw.strip("()")
        return ""
