"""Pure skill composition helpers: sanitize traces, parameterize actions,
render templates, and derive verification contracts. Stateless — extracted from
the orchestrator to keep the loop lean."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from cua_agent.orchestrator.planning import Step

SKILL_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class SkillComposer:
    """Transforms for turning recorded action traces into skills.

    ``platform_name`` is stamped into a skill's preconditions so the fast-path
    filter can match it against the running platform. Left blank, the platform
    precondition is omitted (so a skill stays selectable everywhere).
    """

    def __init__(self, platform_name: str = "") -> None:
        self.platform_name = (platform_name or "").strip()

    def _extract_recovered_actions(self, step_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        last_failure_idx = -1
        for idx, event in enumerate(step_trace):
            if not event.get("success", True):
                last_failure_idx = idx
        candidate_window = step_trace[last_failure_idx + 1 :] if step_trace else []

        extracted: List[Dict[str, Any]] = []
        for event in candidate_window:
            if not event.get("success", True):
                continue
            raw_action = event.get("action") or {}
            if raw_action.get("type") == "macro_actions":
                for sub in raw_action.get("actions") or []:
                    cleaned_sub = self._sanitize_action_for_skill(sub)
                    if cleaned_sub:
                        extracted.append(cleaned_sub)
                continue
            cleaned = self._sanitize_action_for_skill(raw_action)
            if cleaned:
                extracted.append(cleaned)

        deduped: List[Dict[str, Any]] = []
        for act in extracted:
            if deduped and act == deduped[-1]:
                continue
            deduped.append(act)
        return deduped
    def _sanitize_action_for_skill(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed_types = {
            "left_click",
            "right_click",
            "double_click",
            "click_element",
            "mouse_move",
            "hover",
            "drag_and_drop",
            "select_area",
            "scroll",
            "scroll_to_element",
            "type",
            "fill_field",
            "click_and_type",
            "key",
            "open_app",
            "focus_window",
            "wait",
            "wait_for_element",
            "wait_for_idle",
        }
        action_type = action.get("type")
        if action_type not in allowed_types:
            return None

        keep_keys = {
            "type",
            "x",
            "y",
            "target_x",
            "target_y",
            "scroll_y",
            "clicks",
            "axis",
            "text",
            "keys",
            "seconds",
            "duration",
            "hold_delay",
            "app_name",
            "phantom_mode",
            "semantic_role",
            "semantic_label",
            "semantic_path",
            "role",
            "label",
            "path",
            "element_ref",
            "window_title",
            "submit",
            "clear",
            "timeout",
            "max_scrolls",
            "paste",
            "capture_selection",
            "click_type",
        }
        cleaned: Dict[str, Any] = {}
        for key in keep_keys:
            if key in action and action.get(key) is not None:
                cleaned[key] = action.get(key)
        if cleaned.get("type") == "scroll" and "clicks" not in cleaned:
            if action.get("scroll_y") is not None:
                cleaned["clicks"] = int(action.get("scroll_y"))
            else:
                cleaned["clicks"] = 0
        if "type" not in cleaned:
            cleaned["type"] = action_type
        return cleaned
    def _materialize_skill_actions(
        self,
        skill: Any,
        runtime_args: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        parameters = skill.parameters if isinstance(getattr(skill, "parameters", None), dict) else {}
        resolved_args: Dict[str, Any] = {}

        for key, value in (runtime_args or {}).items():
            if self._is_skill_arg_scalar(value):
                resolved_args[str(key)] = value

        missing: List[str] = []
        for param_name, spec in parameters.items():
            token = str(param_name)
            if token in resolved_args and resolved_args[token] not in (None, ""):
                continue

            default_value = self._skill_param_default(spec)
            if default_value is not None:
                resolved_args[token] = default_value
                continue

            if self._skill_param_required(spec):
                missing.append(token)

        rendered = self._render_template_value(skill.actions, resolved_args)
        if not isinstance(rendered, list):
            return [], resolved_args, sorted(set(missing))

        rendered_actions = [dict(item) for item in rendered if isinstance(item, dict)]
        unresolved = self._extract_template_placeholders(rendered_actions)
        unresolved_missing = sorted(name for name in unresolved if name not in resolved_args)
        if unresolved_missing:
            missing.extend(unresolved_missing)

        return rendered_actions, resolved_args, sorted(set(missing))
    def _render_skill_verification_contract(
        self,
        skill: Any,
        resolved_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_contract = (
            skill.verification_contract
            if isinstance(getattr(skill, "verification_contract", None), dict)
            else {}
        )
        if not raw_contract:
            return {}
        rendered = self._render_template_value(raw_contract, resolved_args)
        if not isinstance(rendered, dict):
            return {}
        contract: Dict[str, Any] = {}
        if rendered.get("sensor") is not None:
            contract["sensor"] = str(rendered.get("sensor")).strip().lower()
        if rendered.get("expected_state") is not None:
            expected = str(rendered.get("expected_state")).strip()
            if expected:
                contract["expected_state"] = expected[:500]
        if rendered.get("timeout_seconds") is not None:
            try:
                timeout = int(rendered.get("timeout_seconds"))
                contract["timeout_seconds"] = max(1, min(timeout, 30))
            except (TypeError, ValueError):
                pass
        return contract
    def _build_composable_skill_payload(
        self,
        actions: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        templated_actions = [dict(action) for action in actions if isinstance(action, dict)]
        parameters: Dict[str, Any] = {}
        counters: Dict[str, int] = {}
        param_fields = ("text", "content", "app_name", "url", "value")

        for action in templated_actions:
            for field in param_fields:
                raw_value = action.get(field)
                if not isinstance(raw_value, str):
                    continue
                value = raw_value.strip()
                if not value:
                    continue
                # Keep explicit templates intact.
                if SKILL_PLACEHOLDER_RE.search(value):
                    continue
                if not any(char.isalnum() for char in value):
                    continue

                counters[field] = counters.get(field, 0) + 1
                param_name = f"{field}_{counters[field]}"
                action[field] = "{" + param_name + "}"
                parameters[param_name] = {
                    "description": f"Runtime value for '{field}' in the skill action sequence.",
                    "required": False,
                    "default": raw_value,
                }

        return templated_actions, parameters
    def _derive_skill_verification_contract(
        self,
        current_step: Optional[Step],
        source_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_contract = (
            source_action.get("verification")
            if isinstance(source_action, dict) and isinstance(source_action.get("verification"), dict)
            else {}
        )
        if source_contract:
            contract: Dict[str, Any] = {}
            if source_contract.get("sensor") is not None:
                contract["sensor"] = str(source_contract.get("sensor")).strip().lower()
            if source_contract.get("expected_state") is not None:
                expected = str(source_contract.get("expected_state")).strip()
                if expected:
                    contract["expected_state"] = expected[:500]
            if source_contract.get("timeout_seconds") is not None:
                try:
                    timeout = int(source_contract.get("timeout_seconds"))
                    contract["timeout_seconds"] = max(1, min(timeout, 30))
                except (TypeError, ValueError):
                    pass
            if contract:
                return contract

        if current_step and getattr(current_step, "expected_state", ""):
            expected = str(current_step.expected_state).strip()
            if expected:
                return {
                    "sensor": "a11y_tree",
                    "expected_state": expected[:500],
                    "timeout_seconds": 5,
                }
        return {}
    def _render_template_value(self, value: Any, args: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._render_template_value(val, args) for key, val in value.items()}
        if isinstance(value, list):
            return [self._render_template_value(item, args) for item in value]
        if not isinstance(value, str):
            return value

        exact = SKILL_PLACEHOLDER_RE.fullmatch(value.strip())
        if exact:
            token = exact.group(1)
            if token in args:
                return args[token]

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            replacement = args.get(token)
            if replacement is None:
                return match.group(0)
            return str(replacement)

        return SKILL_PLACEHOLDER_RE.sub(_replace, value)
    def _extract_template_placeholders(self, value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for child in value.values():
                found.update(self._extract_template_placeholders(child))
            return found
        if isinstance(value, list):
            for child in value:
                found.update(self._extract_template_placeholders(child))
            return found
        if isinstance(value, str):
            for match in SKILL_PLACEHOLDER_RE.finditer(value):
                found.add(match.group(1))
        return found
    def _skill_param_default(self, spec: Any) -> Any:
        if isinstance(spec, dict):
            default = spec.get("default")
            if self._is_skill_arg_scalar(default):
                return default
            return None
        if isinstance(spec, (int, float, bool)):
            return spec
        return None
    def _skill_param_required(self, spec: Any) -> bool:
        if isinstance(spec, dict):
            return bool(spec.get("required", False))
        return False
    def _is_skill_arg_scalar(self, value: Any) -> bool:
        return isinstance(value, (str, int, float, bool))
    def _skill_context_metadata(self, actions: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        labels: set[str] = set()
        roles: set[str] = set()
        paths: set[str] = set()
        for act in actions or []:
            for key in ("semantic_label", "label"):
                value = str(act.get(key) or "").strip()
                if value and "{" not in value:
                    labels.add(value[:120])
            for key in ("semantic_role", "role"):
                value = str(act.get(key) or "").strip()
                if value:
                    roles.add(value[:80])
            value = str(act.get("semantic_path") or act.get("path") or "").strip()
            if value and "{" not in value:
                paths.add(value[:180])
        preconditions: Dict[str, Any] = {}
        if self.platform_name:
            preconditions["platform"] = self.platform_name
        grounding_signature = {
            "labels": sorted(labels),
            "roles": sorted(roles),
            "paths": sorted(paths),
        }
        return preconditions, grounding_signature

    def _slugify(self, text: str) -> str:
        """Lightweight slug for skill names."""
        if not text:
            return ""
        lowered = text.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        return slug or "macro"
