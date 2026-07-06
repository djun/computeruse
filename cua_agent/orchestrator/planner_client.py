"""Client for turning a user prompt into a structured step plan."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from cua_agent.memory.memory_manager import Episode, SemanticMemoryItem
from cua_agent.orchestrator.planning import Plan, Step
from cua_agent.utils.config import Settings
from cua_agent.utils.image_mime import configured_image_mime, image_data_uri
from cua_agent.utils.logger import get_logger
from cua_agent.utils.token_usage import usage_tokens


class PlannerClient:
    """Turns a user prompt and prior context into a structured plan."""

    def __init__(self, settings: Settings, platform_name: str = "desktop") -> None:
        self.settings = settings
        self.platform_name = platform_name or "desktop"
        self.logger = get_logger(__name__, level=settings.log_level)
        self.client = self._build_client()
        self.tokens_used = 0

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
                                "preferred_sensor": {
                                    "type": "string",
                                    "enum": ["os_telemetry", "a11y_tree", "pixel_diff", "vision_full"],
                                    "default": "a11y_tree",
                                },
                                "risk_level": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                    "default": "low",
                                },
                                "grounding_strategy": {
                                    "type": "string",
                                    "enum": ["semantic_first", "visual_first", "fusion_required"],
                                    "default": "semantic_first",
                                },
                            },
                            "required": [
                                "id",
                                "description",
                                "success_criteria",
                                "status",
                                "expected_state",
                                "recovery_steps",
                                "sub_steps",
                            ],
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
            # No generic step-level expected_state on purpose: resolve_contract would
            # force it onto every action regardless of the resolved sensor, hard-failing
            # valid actions (e.g. open_app under os_telemetry, where the snapshot is
            # clipboard-only) and bypassing the visual fallback. Leaving it empty lets
            # action-specific defaults and "any"/visual-fallback semantics apply.
            steps = [
                Step(
                    id=0,
                    description="Inspect the desktop and orient to the request",
                    success_criteria="Relevant app or window is visible and ready",
                    status="in_progress",
                ),
                Step(
                    id=1,
                    description=f"Execute the task: {user_prompt}",
                    success_criteria="On-screen confirmation of the completed request (visible result, file, or page)",
                ),
            ]
            return Plan(id=plan_id, user_prompt=user_prompt, steps=steps, current_step_index=0)

        context = self._format_memory_context(prior_episodes or [], prior_semantic or [])
        fallback_mime = configured_image_mime(self.settings.encode_format)
        
        system_prompt = (
            "You are a task planner for a desktop agent. "
            "First, THINK step-by-step about the user request, the current screen state, and potential obstacles. "
            "Then, output a JSON object with an ordered `steps` array.\n"
            "Each step must have: id (int), description (string), success_criteria (string), status (pending|in_progress|done|failed), notes (string), expected_state (string), recovery_steps (array of strings), sub_steps (array of strings), preferred_sensor, risk_level, grounding_strategy.\n"
            "- Split the task into 3-7 small, verifiable steps. Keep main steps HIGH-LEVEL and list concrete clicks/fields in sub_steps.\n"
            "- Apply SMART goal principles.\n"
            "- 'sub_steps': Break complex steps into atomic actions (e.g. 'Click File', 'Select Print').\n"
            "- 'description': Specific and Action-oriented.\n"
            "- 'success_criteria': Measurable and VISUAL.\n"
            "- Mark the first step status as 'in_progress'.\n"
            "Output only the plan JSON object matching the schema."
        )
        
        user_content = [
            {"type": "text", "text": f"User request: {user_prompt}\n\nPrior context:\n{context}"},
        ]
        if screenshot_b64:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_b64, fallback=fallback_mime)}}
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
            self.tokens_used += usage_tokens(response)
            message = response.choices[0].message if response and response.choices else None
            content = message.content if message else "{}"
            plan_dict = self._parse_plan_response(message or content, plan_id, user_prompt)
            return Plan.from_dict(plan_dict)
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Planner call failed; using fallback plan: %s", exc)
            # See the no-client fallback above: avoid generic step-level expected_state
            # so valid actions are not hard-failed by a sensor-incompatible contract.
            steps = [
                Step(
                    id=0,
                    description="Inspect the desktop and orient to the request",
                    success_criteria="Relevant app or window is visible and ready",
                    status="in_progress",
                ),
                Step(
                    id=1,
                    description=f"Execute the task: {user_prompt}",
                    success_criteria="On-screen confirmation of the completed request (visible result, file, or page)",
                ),
            ]
            return Plan(id=plan_id, user_prompt=user_prompt, steps=steps, current_step_index=0)

    def revise_plan(self, plan: Plan, history: List[str], screenshot_b64: str) -> Plan:
        """Ask the planner to refine an in-flight plan based on progress and the current UI."""
        if not self.client:
            self.logger.info("Planner revision skipped: client unavailable.")
            return plan

        fallback_mime = configured_image_mime(self.settings.encode_format)
        system_prompt = (
            f"You are revising an in-flight {self.platform_name} desktop plan. "
            "First, REASON about the failure or current state. "
            "Then, output an UPDATED plan JSON.\n"
            "Schema: id, user_prompt, steps (id, description, success_criteria, status, notes, expected_state, recovery_steps, sub_steps, preferred_sensor, risk_level, grounding_strategy), current_step_index.\n"
            "- Keep 3-7 concise steps.\n"
            "- 'success_criteria' must be VISUAL.\n"
            "- Mark steps as done if satisfied.\n"
            "- Mark blocked steps as failed.\n"
            "- Ensure exactly one step is 'in_progress'.\n"
            "Output only the updated plan JSON object matching the schema."
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
            {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_b64, fallback=fallback_mime)}},
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
            self.tokens_used += usage_tokens(response)
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
                    recovery_steps=self._string_list(raw.get("recovery_steps", [])),
                    sub_steps=self._string_list(raw.get("sub_steps", [])),
                    preferred_sensor=self._enum_value(
                        raw.get("preferred_sensor"),
                        {"os_telemetry", "a11y_tree", "pixel_diff", "vision_full"},
                        "a11y_tree",
                    ),
                    risk_level=self._enum_value(raw.get("risk_level"), {"low", "medium", "high"}, "low"),
                    grounding_strategy=self._enum_value(
                        raw.get("grounding_strategy"),
                        {"semantic_first", "visual_first", "fusion_required"},
                        "semantic_first",
                    ),
                )
                steps.append(step)
            except Exception:
                continue
        if not steps:
            # No generic step-level expected_state: resolve_contract would force
            # it onto every action regardless of the resolved sensor, hard-failing
            # valid actions. Matches the make_plan no-client/failure fallbacks.
            steps = [
                Step(
                    id=0,
                    description="Inspect the desktop and orient to the request",
                    success_criteria="Relevant app or window is visible and ready",
                    status="in_progress",
                ),
                Step(
                    id=1,
                    description=f"Execute the task: {user_prompt}",
                    success_criteria="On-screen confirmation of the completed request (visible result, file, or page)",
                ),
            ]
        plan = Plan(
            id=data.get("id", plan_id),
            user_prompt=data.get("user_prompt", user_prompt),
            steps=steps,
            current_step_index=data.get("current_step_index", 0),
        )
        plan = self._repair_plan_invariants(plan)

        return plan.to_dict()

    def _repair_plan_invariants(self, plan: Plan) -> Plan:
        if not plan.steps:
            return plan

        allowed_status = {"pending", "in_progress", "done", "failed"}
        allowed_sensors = {"os_telemetry", "a11y_tree", "pixel_diff", "vision_full"}
        allowed_risk = {"low", "medium", "high"}
        allowed_grounding = {"semantic_first", "visual_first", "fusion_required"}

        for idx, step in enumerate(plan.steps):
            if step.status not in allowed_status:
                step.status = "pending"
            # Deliberately do NOT backfill expected_state from success_criteria/
            # description: a generic step-level expected_state is forced onto
            # every action by resolve_contract regardless of the resolved sensor,
            # hard-failing valid actions (see the make_plan fallback comment).
            # Action-level expected_effect provides the concrete contract instead.
            step.recovery_steps = self._string_list(step.recovery_steps)
            step.sub_steps = self._string_list(step.sub_steps)
            step.preferred_sensor = self._enum_value(step.preferred_sensor, allowed_sensors, "a11y_tree")
            step.risk_level = self._enum_value(step.risk_level, allowed_risk, "low")
            step.grounding_strategy = self._enum_value(
                step.grounding_strategy, allowed_grounding, "semantic_first"
            )
            if not isinstance(step.id, int):
                step.id = idx

        try:
            current_idx = int(plan.current_step_index)
        except (TypeError, ValueError):
            current_idx = 0
        if current_idx < 0:
            current_idx = 0

        in_progress_indices = [i for i, step in enumerate(plan.steps) if step.status == "in_progress"]
        if in_progress_indices:
            current_idx = in_progress_indices[0]
        elif current_idx >= len(plan.steps) or plan.steps[current_idx].status in {"done", "failed"}:
            # Out-of-range or already-resolved index: repair to the first open step so we
            # never drop step context while steps remain pending. When every step is
            # done/failed, first_open == len(steps), which correctly marks the plan complete.
            current_idx = next(
                (i for i, step in enumerate(plan.steps) if step.status not in {"done", "failed"}),
                len(plan.steps),
            )

        if current_idx < len(plan.steps):
            for idx, step in enumerate(plan.steps):
                if idx < current_idx and step.status in {"pending", "in_progress"}:
                    step.status = "done"
                elif idx == current_idx:
                    step.status = "in_progress"
                elif idx > current_idx and step.status == "in_progress":
                    step.status = "pending"
        plan.current_step_index = current_idx
        return plan

    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _enum_value(self, value: Any, allowed: set[str], default: str) -> str:
        token = str(value or "").strip().lower()
        return token if token in allowed else default

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
            self.tokens_used += usage_tokens(response)
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
            self.tokens_used += usage_tokens(response)
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
