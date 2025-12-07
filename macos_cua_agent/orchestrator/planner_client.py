from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from macos_cua_agent.memory.memory_manager import Episode, SemanticMemoryItem
from macos_cua_agent.orchestrator.planning import Plan, Step
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class PlannerClient:
    """Turns a user prompt and prior context into a structured plan."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.client = self._build_client()

    def _plan_json_schema(self) -> Dict[str, Any]:
        """JSON schema for structured plan outputs."""
        return {
            "name": "desktop_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Plan identifier"},
                    "user_prompt": {"type": "string"},
                    "current_step_index": {"type": "integer"},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "description": {"type": "string"},
                                "success_criteria": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done", "failed"],
                                },
                                "notes": {"type": "string", "default": ""},
                                "expected_state": {"type": "string", "default": ""},
                                "recovery_steps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                                "sub_steps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                            },
                            "required": ["id", "description", "success_criteria", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "user_prompt", "steps", "current_step_index"],
                "additionalProperties": False,
            },
        }

    def _build_client(self) -> Optional[Any]:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self.logger.warning("openai package unavailable for planner: %s", exc)
            return None

        if not self.settings.planner_api_key:
            self.logger.warning("Planner key missing; set OPENROUTER_API_KEY (or PLANNER_API_KEY override) for planning.")
            return None

        return OpenAI(base_url=self.settings.planner_base_url, api_key=self.settings.planner_api_key)

    def make_plan(
        self,
        user_prompt: str,
        prior_episodes: List[Episode] | None = None,
        prior_semantic: List[SemanticMemoryItem] | None = None,
        screenshot_b64: str | None = None,
    ) -> Plan:
        plan_id = str(uuid.uuid4())
        if not self.client:
            steps = [
                Step(id=0, description="Inspect the desktop and orient to the request", success_criteria="Relevant app or window is visible and ready", status="in_progress"),
                Step(id=1, description=f"Execute the task: {user_prompt}", success_criteria="On-screen confirmation of the completed request (visible result, file, or page)"),
            ]
            return Plan(id=plan_id, user_prompt=user_prompt, steps=steps, current_step_index=0)

        context = self._format_memory_context(prior_episodes or [], prior_semantic or [])
        mime = "image/png" if self.settings.encode_format.lower() == "png" else "image/jpeg"
        
        system_prompt = (
            "You are a task planner for a macOS desktop agent. "
            "First, THINK step-by-step about the user request, the current screen state, and potential obstacles. "
            "Then, output a JSON object with an ordered `steps` array.\n"
            "Each step must have: id (int), description (string), success_criteria (string), status (pending|in_progress|done|failed), notes (string), expected_state (string), recovery_steps (array of strings), sub_steps (array of strings).\n"
            "- Split the task into 3-7 small, verifiable steps. Keep main steps HIGH-LEVEL and list concrete clicks/fields in sub_steps.\n"
            "- Apply SMART goal principles.\n"
            "- 'sub_steps': Break complex steps into atomic actions (e.g. 'Click File', 'Select Print').\n"
            "- 'description': Specific and Action-oriented.\n"
            "- 'success_criteria': Measurable and VISUAL.\n"
            "- Mark the first step status as 'in_progress'.\n"
            "Output format: \n"
            "REASONING: <your thought process>\n"
            "PLAN_JSON: <the valid JSON object>"
        )
        
        user_content = [
            {"type": "text", "text": f"User request: {user_prompt}\n\nPrior context:\n{context}"},
        ]
        if screenshot_b64:
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{screenshot_b64}"}}
            )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.planner_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_schema", "json_schema": self._plan_json_schema()},
                extra_body={"structured_outputs": {"type": "json_schema", "json_schema": self._plan_json_schema()}},
            )
            message = response.choices[0].message if response and response.choices else None
            content = message.content if message else "{}"
            plan_dict = self._parse_plan_response(message or content, plan_id, user_prompt)
            return Plan.from_dict(plan_dict)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Planner call failed; using fallback plan: %s", exc)
            steps = [
                Step(id=0, description="Inspect the desktop and orient to the request", success_criteria="Relevant app or window is visible and ready", status="in_progress"),
                Step(id=1, description=f"Execute the task: {user_prompt}", success_criteria="On-screen confirmation of the completed request (visible result, file, or page)"),
            ]
            return Plan(id=plan_id, user_prompt=user_prompt, steps=steps, current_step_index=0)

    def revise_plan(self, plan: Plan, history: List[str], screenshot_b64: str) -> Plan:
        """Ask the planner to refine an in-flight plan based on progress and the current UI."""
        if not self.client:
            self.logger.info("Planner revision skipped: client unavailable.")
            return plan

        mime = "image/png" if self.settings.encode_format.lower() == "png" else "image/jpeg"
        system_prompt = (
            "You are revising an in-flight macOS desktop plan. "
            "First, REASON about the failure or current state. "
            "Then, output an UPDATED plan JSON.\n"
            "Schema: id, user_prompt, steps (id, description, success_criteria, status, notes, expected_state, recovery_steps, sub_steps), current_step_index.\n"
            "- Keep 3-7 concise steps.\n"
            "- 'success_criteria' must be VISUAL.\n"
            "- Mark steps as done if satisfied.\n"
            "- Mark blocked steps as failed.\n"
            "- Ensure exactly one step is 'in_progress'.\n"
            "Output format: \n"
            "REASONING: <text>\n"
            "PLAN_JSON: <json>"
        )
        plan_json = json.dumps(plan.to_dict())
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Existing plan:\n{plan_json}\n\nRecent events (most recent last):\n"
                    + "\n".join(history[-40:])
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{screenshot_b64}"}},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.settings.planner_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_schema", "json_schema": self._plan_json_schema()},
                extra_body={"structured_outputs": {"type": "json_schema", "json_schema": self._plan_json_schema()}},
            )
            message = response.choices[0].message if response and response.choices else None
            content = message.content if message else "{}"
            plan_dict = self._parse_plan_response(message or content, plan.id, plan.user_prompt)
            return Plan.from_dict(plan_dict)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Plan revision failed; keeping existing plan: %s", exc)
            return plan

    def _parse_plan_response(self, content: Any, plan_id: str, user_prompt: str) -> Dict[str, Any]:
        # Structured output path
        parsed = getattr(content, "parsed", None)
        if parsed:
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                data = parsed
            else:
                data = {}
        else:
            raw_content = getattr(content, "content", content)
            if isinstance(raw_content, list):
                # Attempt to stitch together parts from content fragments
                part_texts = []
                json_candidates = []
                for frag in raw_content:
                    if isinstance(frag, dict):
                        if "json" in frag and isinstance(frag["json"], dict):
                            json_candidates.append(frag["json"])
                        elif "text" in frag:
                            part_texts.append(str(frag["text"]))
                    elif hasattr(frag, "text"):
                        part_texts.append(str(frag.text))  # type: ignore
                if json_candidates:
                    data = json_candidates[0]
                else:
                    raw_content = "".join(part_texts)

            if not isinstance(raw_content, str):
                raw_content = json.dumps(raw_content or {})
            
            json_str = raw_content
            if "PLAN_JSON:" in raw_content:
                parts = raw_content.split("PLAN_JSON:", 1)
                if len(parts) > 1:
                    json_str = parts[1].strip()
            else:
                # Fallback: try to find the first brace
                start = raw_content.find("{")
                end = raw_content.rfind("}")
                if start != -1 and end != -1:
                    json_str = raw_content[start : end + 1]

            try:
                data = json.loads(json_str)
            except Exception:
                snippet = raw_content if isinstance(raw_content, str) else str(raw_content)
                self.logger.warning("Failed to parse plan JSON from content: %s", snippet[:200])
                data = {}

        raw_steps = data.get("steps") or []
        steps: List[Step] = []
        for idx, raw in enumerate(raw_steps):
            try:
                step = Step(
                    id=int(raw.get("id", idx)),
                    description=str(raw.get("description", "")).strip() or f"Step {idx + 1}",
                    success_criteria=str(raw.get("success_criteria", "")).strip()
                    or "Criteria not provided",
                    status=str(raw.get("status", "pending")),
                    notes=str(raw.get("notes", "")),
                    expected_state=str(raw.get("expected_state", "")),
                    recovery_steps=raw.get("recovery_steps", []),
                    sub_steps=raw.get("sub_steps", []),
                )
                steps.append(step)
            except Exception:
                continue
        if not steps:
            steps = [
                Step(id=0, description="Inspect the desktop and orient to the request", success_criteria="Relevant app or window is visible and ready", status="in_progress"),
                Step(id=1, description=f"Execute the task: {user_prompt}", success_criteria="On-screen confirmation of the completed request (visible result, file, or page)"),
            ]
        if steps and steps[0].status == "pending":
            steps[0].status = "in_progress"

        return {
            "id": data.get("id", plan_id),
            "user_prompt": data.get("user_prompt", user_prompt),
            "steps": [s.to_dict() for s in steps],
            "current_step_index": data.get("current_step_index", 0),
        }

    def _format_memory_context(self, episodes: List[Episode], semantic_items: List[SemanticMemoryItem]) -> str:
        chunks: List[str] = []
        if episodes:
            for ep in episodes[-3:]:
                chunks.append(
                    f"- Episode {ep.id}: prompt='{ep.user_prompt[:60]}', outcome={ep.outcome}, summary={ep.summary}"
                )
        if semantic_items:
            for item in semantic_items[:5]:
                chunks.append(f"- Semantic note {item.id}: {item.text[:120]}")
        return "\n".join(chunks) if chunks else "No prior memory available."

    def summarize_episode(self, user_prompt: str, history: List[str], plan: Optional[Plan] = None) -> str:
        if not self.client:
            return self._fallback_summary(history)

        plan_line = ""
        if plan:
            step_bits = [f"{s.id}:{s.status}" for s in plan.steps]
            plan_line = f" Plan steps: {'; '.join(step_bits)}"

        trimmed_history = "\n".join(history[-80:])
        system_prompt = (
            "Summarize the desktop control session in 2-4 sentences. "
            "Highlight what was attempted, what worked, and outstanding blockers. "
            "Do not include tool call JSON; keep it high level."
        )
        user_block = f"User prompt: {user_prompt}.{plan_line}\n\nRecent events:\n{trimmed_history}"
        try:
            response = self.client.chat.completions.create(
                model=self.settings.planner_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_block},
                ],
            )
            content = response.choices[0].message.content if response and response.choices else ""
            if isinstance(content, list):
                content = "".join([frag.text for frag in content if hasattr(frag, "text")])  # type: ignore
            return str(content or "").strip() or self._fallback_summary(history)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Planner summary failed: %s", exc)
            return self._fallback_summary(history)

    def summarize_history_chunk(self, history_chunk: List[str]) -> str:
        """Compress a list of history events into a single summary line."""
        if not self.client or not history_chunk:
            return ""
        
        text_block = "\n".join(history_chunk)
        system_prompt = (
            "Compress the following list of agent events into a single concise summary sentence. "
            "Focus on actions taken and their outcomes. Ignore noise."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.planner_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_block[:4000]},  # Safe truncation
                ],
                max_tokens=200,
            )
            content = response.choices[0].message.content if response and response.choices else ""
            return str(content or "").strip()
        except Exception:
            return ""

    def _fallback_summary(self, history: List[str]) -> str:
        if not history:
            return "No actions recorded."
        head = history[:3]
        tail = history[-3:] if len(history) > 3 else []
        snippet = " | ".join(head + tail)
        return f"Session summary unavailable; raw history snippet: {snippet}"
