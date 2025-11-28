from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from macos_cua_agent.orchestrator.planning import Step
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


@dataclass
class ReflectionResult:
    is_complete: bool
    status: str  # success|incomplete|failed
    failure_type: str = ""  # e.g., blocked, visual_mismatch, timeout
    reason: str = ""


class Reflector:
    """Uses a secondary model (ChatGPT/Gemini/etc.) to verify progress or offer hints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.client = self._build_client()
        self.mime = "image/png" if settings.encode_format.lower() == "png" else "image/jpeg"

    def _build_client(self) -> Optional[object]:
        if not self.settings.enable_reflection:
            return None
        api_key = self.settings.reflector_api_key
        if not api_key:
            self.logger.info("Reflection disabled: REFLECTOR_API_KEY/OPENROUTER_API_KEY missing.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self.logger.warning("openai package unavailable for reflection: %s", exc)
            return None
        return OpenAI(base_url=self.settings.reflector_base_url, api_key=api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def evaluate_step(self, step: Step, history: List[str], screenshot_b64: str, changed: bool) -> ReflectionResult:
        """Ask the verifier model if the step is complete and diagnose issues."""
        if not self.client:
            return ReflectionResult(is_complete=False, status="incomplete", reason="Reflection client unavailable")

        expected_text = ""
        if getattr(step, "expected_state", ""):
            expected_text = f"Expected state before the step: {step.expected_state}\n"
        
        prompt = (
            "You are a visual verifier for a desktop agent. Given the current sub-step, "
            "recent history, and a screenshot, evaluate the progress.\n"
            "Step: {desc}\nSuccess criteria: {criteria}\n{expected_text}\n"
            "Respond in JSON format with:\n"
            "- is_complete (bool): true if strictly satisfied.\n"
            "- status (str): 'success', 'incomplete', or 'failed' (if clearly blocked or erroneous).\n"
            "- failure_type (str): 'visual_mismatch', 'blocked_by_popup', 'no_change', 'error_message', or empty if success.\n"
            "- reason (str): concise explanation."
        ).format(desc=step.description, criteria=step.success_criteria, expected_text=expected_text)

        content = [
            {"type": "text", "text": f"{prompt}\n\nRecent events:\n" + "\n".join(history[-20:])},
            {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{screenshot_b64}"}},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.settings.reflector_model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": content},
                ],
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content if response and response.choices else "{}"
            import json
            data = json.loads(raw or "{}")
            
            return ReflectionResult(
                is_complete=data.get("is_complete", False),
                status=data.get("status", "incomplete"),
                failure_type=data.get("failure_type", ""),
                reason=data.get("reason", "")
            )
        except Exception as exc:
            self.logger.warning("Reflection check failed: %s", exc)
            return ReflectionResult(is_complete=False, status="error", reason=str(exc))

    # Legacy compatibility wrapper if needed, or just remove if we update calls.
    def is_step_complete(self, step: Step, history: List[str], screenshot_b64: str, changed: bool) -> bool:
        result = self.evaluate_step(step, history, screenshot_b64, changed)
        return result.is_complete

    def suggest_hint(self, step: Optional[Step], history: List[str], screenshot_b64: str) -> str:
        if not self.client:
            return ""
        step_text = ""
        if step:
            step_text = f"Current step: {step.description}. Success criteria: {step.success_criteria}."
        prompt = (
            "The desktop agent appears stuck. Based on the screenshot and recent events, "
            "give one short hint (<=20 words) to unblock progress. Avoid telling the user, "
            "just give the agent a next idea."
        )
        content = [
            {"type": "text", "text": f"{prompt}\n{step_text}\nRecent events:\n" + "\n".join(history[-30:])},
            {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{screenshot_b64}"}},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.reflector_model,
                messages=[
                    {"role": "system", "content": "Provide one concise hint, no preamble."},
                    {"role": "user", "content": content},
                ],
                max_tokens=1000,
            )
            raw = response.choices[0].message.content if response and response.choices else ""
            return "".join([frag.text for frag in raw]) if isinstance(raw, list) else str(raw or "")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Reflection hint failed: %s", exc)
            return ""

    def describe_image(self, screenshot_b64: str) -> str:
        """Generate a concise text description of the screenshot for semantic memory."""
        if not self.client:
            return ""
        
        prompt = (
            "Describe the current desktop state concisely (1-2 sentences). "
            "Focus on the active window, visible applications, and any open documents or websites. "
            "Ignore background wallpaper or system tray unless relevant."
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{screenshot_b64}"}},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.reflector_model,
                messages=[
                    {"role": "system", "content": "Be concise and objective."},
                    {"role": "user", "content": content},
                ],
                max_tokens=500,
            )
            raw = response.choices[0].message.content if response and response.choices else ""
            return "".join([frag.text for frag in raw]) if isinstance(raw, list) else str(raw or "")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Image description failed: %s", exc)
            return ""
