"""Autonomy and target guardrails for proposed actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cua_agent.agent.state_manager import ActionResult, StateManager
from cua_agent.orchestrator.react_types import GroundingBundle
from cua_agent.utils.config import Settings


@dataclass
class ActionPolicyDecision:
    allowed: bool
    action: dict[str, Any]
    risk_level: str = "low"
    reason: str = ""
    target_gid: str | None = None
    target_confidence: float = 0.0


class ActionPolicy:
    """Applies lightweight autonomy rules before adapter execution."""

    HIGH_RISK_TYPES = {"sandbox_shell", "script_op", "browser_op"}
    MEDIUM_RISK_TYPES = {"macro_actions", "drag_and_drop", "select_area", "open_app", "focus_window"}
    TARGETED_TYPES = {
        "left_click",
        "right_click",
        "double_click",
        "click_element",
        "fill_field",
        "click_and_type",
        "hover",
        "type",
    }
    # Risk lives in the target's meaning, not only in the action type: clicking
    # "Delete account" is not the same as clicking "Search".
    ACTIVATION_TYPES = {"left_click", "right_click", "double_click", "click_element", "click_and_type"}
    TEXT_ENTRY_TYPES = {"type", "fill_field", "click_and_type"}
    DESTRUCTIVE_LABEL_TOKENS = (
        "delete",
        "remove",
        "erase",
        "format",
        "uninstall",
        "wipe",
        "destroy",
        "shut down",
        "shutdown",
        "empty trash",
        "factory reset",
        "excluir",
        "apagar",
        "remover",
        "formatar",
        "desinstalar",
        "esvaziar lixeira",
    )
    SECURE_FIELD_ROLE_TOKENS = ("securetextfield", "passwordbox", "password")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def normalize_and_guard(
        self,
        action: dict[str, Any],
        *,
        grounding: GroundingBundle | None,
        state: StateManager | None = None,
        step_risk: str | None = None,
    ) -> ActionPolicyDecision:
        normalized = dict(action or {})
        risk_level = self._risk_level(normalized, step_risk)
        target_gid, target_confidence = self._annotate_target(normalized, grounding)

        if target_gid:
            normalized.setdefault("target_gid", target_gid)
            normalized.setdefault("grounding_confidence", target_confidence)

        sensitive_reason = self._sensitive_target_reason(normalized, grounding)
        if sensitive_reason and self._RISK_ORDER.get(risk_level, 0) < self._RISK_ORDER["high"]:
            risk_level = "high"

        risk_detail = f"{risk_level}-risk action" + (f" ({sensitive_reason})" if sensitive_reason else "")
        block, hitl_required = self._autonomy_gate(risk_level)
        if block:
            return ActionPolicyDecision(
                allowed=False,
                action=normalized,
                risk_level=risk_level,
                reason=f"autonomy policy requires confirmation for {risk_detail}",
                target_gid=target_gid,
                target_confidence=target_confidence,
            )
        if hitl_required:
            # Cannot hard-block (HITL is enabled), but the platform policy engine
            # would not otherwise prompt for policy-classified risk types such as
            # open_app/focus_window/browser_op/macro_actions. Flag the action so the
            # platform action engine routes it through the interactive confirmation.
            normalized["requires_hitl_confirmation"] = True
            normalized["hitl_reason"] = f"autonomy policy: {risk_detail} needs confirmation"

        # Low-confidence target detection covers the top-level action and, for
        # macro wrappers, each targeted subaction (which the orchestrator resolves
        # independently). Any weak target triggers a grounding refresh before acting.
        needs_fresh = self._target_confidence_too_low(normalized, target_confidence)
        if str(normalized.get("type") or "").strip().lower() == "macro_actions":
            for sub in normalized.get("actions") or []:
                if not isinstance(sub, dict):
                    continue
                sub_gid, sub_confidence = self._annotate_target(sub, grounding)
                if sub_gid:
                    sub.setdefault("target_gid", sub_gid)
                    sub.setdefault("grounding_confidence", sub_confidence)
                if self._target_confidence_too_low(sub, sub_confidence):
                    needs_fresh = True
        if needs_fresh:
            normalized["needs_fresh_grounding"] = True

        return ActionPolicyDecision(
            allowed=True,
            action=normalized,
            risk_level=risk_level,
            reason="allowed",
            target_gid=target_gid,
            target_confidence=target_confidence,
        )

    def blocked_result(self, decision: ActionPolicyDecision) -> ActionResult:
        return ActionResult(
            success=False,
            reason=decision.reason or "blocked by action policy",
            metadata={
                "risk_level": decision.risk_level,
                "target_gid": decision.target_gid,
                "target_confidence": decision.target_confidence,
            },
            code="policy_blocked",
            category="policy",
            retryable=False,
            suggested_next=["choose a lower-risk action", "ask_user for confirmation"],
        )

    def should_force_visual(self, state: StateManager, turn_index: int) -> bool:
        interval = max(0, int(getattr(self.settings, "force_visual_every_n_turns", 0) or 0))
        if interval and turn_index > 0 and turn_index % interval == 0:
            return True
        if state.last_grounding and state.last_grounding.quality.get("stale_hash"):
            return True
        return False

    _RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

    def _risk_level(self, action: dict[str, Any], step_risk: str | None) -> str:
        # Step risk (from model/planner) may escalate the intrinsic action risk
        # but must never downgrade it. Otherwise a high-risk action type inside a
        # low-risk step (step risk defaults to "low") would bypass the autonomy
        # guard before HIGH_RISK_TYPES is ever consulted.
        intrinsic = self._intrinsic_risk_level(action)
        step_risk_norm = str(step_risk or "").strip().lower()
        if step_risk_norm in self._RISK_ORDER:
            return max(intrinsic, step_risk_norm, key=lambda level: self._RISK_ORDER[level])
        return intrinsic

    def _intrinsic_risk_level(self, action: dict[str, Any]) -> str:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type in self.HIGH_RISK_TYPES:
            return "high"
        if action_type == "macro_actions":
            sub_risks = [
                self._intrinsic_risk_level(sub)
                for sub in action.get("actions") or []
                if isinstance(sub, dict)
            ]
            if "high" in sub_risks:
                return "high"
            if "medium" in sub_risks:
                return "medium"
        if action_type in self.MEDIUM_RISK_TYPES:
            return "medium"
        return "low"

    def _autonomy_gate(self, risk_level: str) -> tuple[bool, bool]:
        """Return (block, hitl_required) for the given risk level.

        When the autonomy level requires confirmation for this risk, we either hard
        block it (HITL disabled, so confirmation is impossible) or flag it for an
        interactive HITL prompt (HITL enabled). Returning hitl_required keeps the
        autonomy contract honest for risk types the platform policy would not prompt.
        """
        autonomy = str(getattr(self.settings, "autonomy_level", "confirm_risky") or "").strip().lower()
        if autonomy == "supervised":
            needs_confirm = risk_level in {"medium", "high"}
        elif autonomy == "confirm_risky":
            needs_confirm = risk_level == "high"
        else:  # fully_autonomous or unknown
            needs_confirm = False
        if not needs_confirm:
            return False, False
        if bool(self.settings.enable_hitl_prompt):
            return False, True
        return True, False

    def _sensitive_target_reason(self, action: dict[str, Any], grounding: GroundingBundle | None) -> str:
        """Escalation reason when the resolved target is destructive or sensitive."""
        action_type = str(action.get("type") or "").strip().lower()
        if action_type == "macro_actions":
            for sub in action.get("actions") or []:
                if isinstance(sub, dict):
                    reason = self._sensitive_target_reason(sub, grounding)
                    if reason:
                        return reason
            return ""
        if action_type not in self.TARGETED_TYPES:
            return ""

        label, role = self._target_label_role(action, grounding)
        if action_type in self.TEXT_ENTRY_TYPES and role:
            role_l = role.lower()
            if any(token in role_l for token in self.SECURE_FIELD_ROLE_TOKENS):
                return f"text entry into secure field (role={role})"
        if action_type in self.ACTIVATION_TYPES and label:
            label_l = label.lower()
            for token in self.DESTRUCTIVE_LABEL_TOKENS:
                if token in label_l:
                    return f"destructive target label '{label}'"
        return ""

    def _target_label_role(self, action: dict[str, Any], grounding: GroundingBundle | None) -> tuple[str, str]:
        label = str(action.get("semantic_label") or "")
        role = str(action.get("semantic_role") or "")
        if label or role:
            return label, role
        ref = action.get("element_id") if action.get("element_id") is not None else action.get("element_ref")
        token = str(ref).strip() if ref is not None else ""
        if not token:
            return "", ""
        for tag in (grounding.som_tags if grounding else None) or []:
            if token in {str(tag.get("id")), str(tag.get("gid"))}:
                return str(tag.get("label") or ""), str(tag.get("role") or "")
        # A non-numeric ref is itself the human-readable label the model targeted
        # (clean-API target.label), even when no overlay tag matches it.
        if not token.isdigit():
            return token, ""
        return "", ""

    def _annotate_target(
        self, action: dict[str, Any], grounding: GroundingBundle | None
    ) -> tuple[str | None, float]:
        if not grounding:
            return None, 0.0
        tags = grounding.som_tags or []
        target_ref = action.get("element_id") if action.get("element_id") is not None else action.get("element_ref")
        if target_ref is not None:
            token = str(target_ref).strip()
            for tag in tags:
                if token and token in {str(tag.get("id")), str(tag.get("gid"))}:
                    return str(tag.get("gid") or ""), float(tag.get("confidence") or 0.0)
        action_gid = str(action.get("target_gid") or "").strip()
        if action_gid:
            for tag in tags:
                if str(tag.get("gid") or "") == action_gid:
                    return action_gid, float(tag.get("confidence") or 0.0)
        return None, 0.0

    def _target_confidence_too_low(self, action: dict[str, Any], target_confidence: float) -> bool:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type not in self.TARGETED_TYPES:
            return False
        if target_confidence <= 0:
            return False
        return target_confidence < float(getattr(self.settings, "min_grounding_confidence", 0.55))
