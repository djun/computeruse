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
    ) -> Plan:
        plan_id = str(uuid.uuid4())
        if not self.client:
            steps = [
                Step(id=0, description="Inspect the desktop and orient to the request", success_criteria="Relevant app or window is visible and ready", status="in_progress"),
                Step(id=1, description=f"Execute the task: {user_prompt}", success_criteria="On-screen confirmation of the completed request (visible result, file, or page)"),
            ]
            return Plan(id=plan_id, user_prompt=user_prompt, steps=steps, current_step_index=0)

        context = self._format_memory_context(prior_episodes or [], prior_semantic or [])
        system_prompt = (
            "You are a task planner for a macOS desktop agent. "
            "Write a concise JSON object with an ordered `steps` array. "
            "Each step must have fields: id (int), description (string), success_criteria (string), status (pending|in_progress|done|failed), notes (string), expected_state (string), recovery_steps (array of strings).\n"
            "- Split the task into 3-7 small UI-manipulation steps that can usually be finished in 10-60 seconds.\n"
            "- Apply SMART goal principles (Specific, Measurable, Achievable, Relevant, Time-bound) to each step.\n"
            "- 'description' must be Specific and Action-oriented (e.g., 'Click the Search icon', not 'Find file').\n"
            "- 'success_criteria' must be Measurable and VISUAL (e.g., 'Search bar appears', not 'Search is ready').\n"
            "- 'recovery_steps' should list 1-2 alternative actions if the primary method fails (e.g., 'Use Cmd+F instead of menu').\n"
            "- Mark the first step status as 'in_progress'; others should start as 'pending'.\n"
            "- expected_state is optional; use it to note the anticipated UI state before running the step.\n"
            "Example step:\n"
            "{ \"id\": 0, \"description\": \"Open Safari\", \"success_criteria\": \"Safari window is visible\", \"status\": \"in_progress\", \"notes\": \"\", \"expected_state\": \"Dock shows Safari not active\", \"recovery_steps\": [\"Click Finder > Applications > Safari\"] }\n"
            "Do not include any text outside the JSON."
        )
        user_prompt_block = f"User request: {user_prompt}\n\nPrior context:\n{context}"
        try:
            response = self.client.chat.completions.create(
                model=self.settings.planner_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_block},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response and response.choices else "{}"
            plan_dict = self._parse_plan_json(content, plan_id, user_prompt)
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
            "You are revising an in-flight macOS desktop plan. Given the existing plan JSON, "
            "recent events, and a screenshot, return an UPDATED plan JSON. Keep the same schema "
            "as the planner output: id, user_prompt, steps (with id, description, success_criteria, "
            "status, notes, expected_state, recovery_steps), current_step_index.\n"
            "- Keep 3-7 concise, UI-grounded steps that a human could finish in 10-60 seconds each.\n"
            "- Ensure steps follow SMART principles (Specific, Measurable, Achievable, Relevant, Time-bound).\n"
            "- 'success_criteria' must be VISUAL and Measurable.\n"
            "- 'recovery_steps' should list 1-2 alternative actions if the step fails.\n"
            "- Mark steps as done if their success_criteria are visibly satisfied in the screenshot.\n"
            "- Mark obviously blocked steps as failed with a short note; add missing steps if needed.\n"
            "- Ensure exactly one step is 'in_progress' (the first not-done step). Others pending or done.\n"
            "- Do not invent unrelated tasks; align strictly with the user_prompt.\n"
            "Return JSON only."
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
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response and response.choices else "{}"
            plan_dict = self._parse_plan_json(content, plan.id, plan.user_prompt)
            return Plan.from_dict(plan_dict)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Plan revision failed; keeping existing plan: %s", exc)
            return plan

    def _parse_plan_json(self, content: Any, plan_id: str, user_prompt: str) -> Dict[str, Any]:
        if isinstance(content, list):
            content = "".join([frag.text for frag in content if hasattr(frag, "text")])  # type: ignore
        if not isinstance(content, str):
            content = json.dumps(content or {})
        try:
            data = json.loads(content)
        except Exception:
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
