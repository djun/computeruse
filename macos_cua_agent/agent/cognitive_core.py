from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import get_display_info, get_system_info

if TYPE_CHECKING:  # pragma: no cover - typing only
    from macos_cua_agent.orchestrator.planning import Plan, Step

# OpenRouter exposes an OpenAI-compatible tool-calling API. We define our own
# computer tool schema so Claude Opus 4.5 can drive local actions.
COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Control the macOS desktop: move and click the mouse, type, press hotkeys, "
            "scroll, wait, request a screenshot, or inspect the UI tree."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "move_mouse",
                        "left_click",
                        "right_click",
                        "double_click",
                        "drag_and_drop",
                        "select_area",
                        "hover",
                        "probe_ui",
                        "clipboard_op",
                        "run_skill",
                        "scroll",
                        "type",
                        "hotkey",
                        "wait",
                        "screenshot",
                        "open_app",
                        "inspect_ui",
                    ],
                },
                "actions": {
                    "type": "array",
                    "description": "Batch of low-level actions to execute sequentially (macro_actions). Each item mirrors the single-action schema.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "element_id": {"type": "integer"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "target_x": {"type": "number"},
                            "target_y": {"type": "number"},
                            "scroll_y": {"type": "number"},
                            "axis": {"type": "string", "enum": ["vertical", "horizontal"]},
                            "radius": {"type": "number"},
                            "text": {"type": "string"},
                            "app_name": {"type": "string"},
                            "keys": {"type": "array", "items": {"type": "string"}},
                            "seconds": {"type": "number"},
                            "duration": {"type": "number"},
                            "hold_delay": {"type": "number"},
                            "sub_action": {"type": "string", "enum": ["read", "write", "clear"]},
                            "content": {"type": "string"},
                            "phantom_mode": {"type": "boolean"},
                            "verify_after": {"type": "boolean"},
                            "skill_id": {"type": "string"},
                            "skill_name": {"type": "string"},
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
                "x": {"type": "number", "description": "X coordinate in logical display points (after downscaling)."},
                "y": {"type": "number", "description": "Y coordinate in logical display points (after downscaling)."},
                "target_x": {"type": "number", "description": "Destination X for drag_and_drop."},
                "target_y": {"type": "number", "description": "Destination Y for drag_and_drop."},
                "scroll_y": {
                    "type": "number",
                    "description": "Scroll amount (positive up/left, negative down/right).",
                },
                "axis": {
                    "type": "string",
                    "enum": ["vertical", "horizontal"],
                    "default": "vertical",
                    "description": "Scroll axis (vertical or horizontal).",
                },
                "radius": {
                    "type": "number",
                    "description": "Radius (in logical points) for probe_ui to include nearby elements.",
                },
                "text": {"type": "string", "description": "Text to type."},
                "app_name": {"type": "string", "description": "Name of the application to open (for 'open_app' action)."},
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hotkey combo, e.g. ['ctrl','s'].",
                },
                "element_id": {
                    "type": "integer",
                    "description": "ID from the numbered overlay tag. Prefer this over raw coordinates when marks are present.",
                },
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait for the 'wait' action.",
                },
                "duration": {
                    "type": "number",
                    "description": "Duration for hover or drag_and_drop in seconds.",
                },
                "hold_delay": {
                    "type": "number",
                    "description": "Delay before starting drag (mouse hold time).",
                },
                "sub_action": {
                    "type": "string",
                    "enum": ["read", "write", "clear"],
                    "description": "Sub-action for clipboard_op.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to clipboard.",
                },
                "phantom_mode": {
                    "type": "boolean",
                    "description": "If true, try to use AX API (AXPress) without moving physical mouse.",
                },
                "verify_after": {
                    "type": "boolean",
                    "description": "If false, skip post-action verification delay and change-detection capture.",
                    "default": True,
                },
                "skill_id": {
                    "type": "string",
                    "description": "ID of a stored procedural skill to execute (run_skill).",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of a stored procedural skill to execute (run_skill).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": (
            "Run safe, sandboxed shell commands in a constrained workspace. "
            "Use this for local file operations or running short scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Full command line, e.g. 'ls -la' or 'python script.py'.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional relative working directory under the agent workspace.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

NOTEBOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "notebook",
        "description": "Manage a persistent notebook for storing research notes, facts, and data across steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add_note", "clear_notes"]},
                "content": {"type": "string", "description": "The note content to save."},
                "source": {"type": "string", "description": "Source of the info (e.g. url or 'user')."}
            },
            "required": ["action"]
        }
    }
}

BROWSER_TOOL = {
    "type": "function",
    "function": {
        "name": "browser",
        "description": "Interact with web browsers (Safari/Chrome) semantically to read content and get links without OCR.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": ["get_page_content", "get_links", "navigate"]},
                "app_name": {"type": "string", "description": "Safari or Google Chrome", "default": "Safari"},
                "url": {"type": "string", "description": "URL to navigate to"}
            },
            "required": ["command"]
        }
    }
}


class CognitiveCore:
    """Calls Claude Opus 4.5 via OpenRouter with a custom computer tool."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display = get_display_info()
        self.system_info = get_system_info()
        self.client = self._build_client()

    def _build_client(self) -> Optional[Any]:
        if not self.settings.use_openrouter:
            self.logger.info("OpenRouter disabled; running in deterministic stub mode.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.logger.warning("openai package not installed; falling back to stubbed actions.")
            return None

        if not self.settings.openrouter_api_key:
            self.logger.warning("OPENROUTER_API_KEY missing; running stub mode.")
            return None

        return OpenAI(base_url=self.settings.openrouter_base_url, api_key=self.settings.openrouter_api_key)

    def propose_action(
        self,
        observation_b64: str,
        history: List[str],
        user_prompt: Optional[str] = None,
        repeat_info: Optional[Dict[str, Any]] = None,
        plan: Optional["Plan"] = None,
        current_step: Optional["Step"] = None,
        loop_state: Optional[Dict[str, Any]] = None,
        ax_tree: Optional[Dict[str, Any]] = None,
        som_tags: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return the next action as a dict with at least a `type` field."""
        if not self.client:
            action_type = "noop" if history else "capture_only"
            return {
                "type": action_type,
                "reason": "Cognitive core running without OpenRouter client.",
            }

        try:
            response = self._call_openrouter(
                observation_b64,
                history,
                user_prompt=user_prompt,
                repeat_info=repeat_info,
                plan=plan,
                current_step=current_step,
                loop_state=loop_state,
                ax_tree=ax_tree,
                som_tags=som_tags,
            )
            parsed_action = self._parse_tool_call(response)
            if parsed_action:
                return parsed_action
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("OpenRouter call failed; falling back to noop.", exc_info=exc)

        return {"type": "noop", "reason": "Failed to generate action"}

    def _call_openrouter(
        self,
        observation_b64: str,
        history: List[str],
        user_prompt: Optional[str],
        repeat_info: Optional[Dict[str, Any]],
        plan: Optional["Plan"],
        current_step: Optional["Step"],
        loop_state: Optional[Dict[str, Any]],
        ax_tree: Optional[Dict[str, Any]],
        som_tags: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Send a vision + tool-calling request to OpenRouter."""
        plan_text = "No structured plan; infer progress from the user's request."
        if plan and current_step:
            upcoming = [
                f"- Step {s.id}: {s.description} (status={s.status})"
                for s in plan.steps
                if s.id != current_step.id
            ]
            plan_text = (
                "Current goal:\n"
                f"- Step {current_step.id}: {current_step.description}\n"
                f"- Success criteria: {current_step.success_criteria}\n"
            )
            if getattr(current_step, "expected_state", ""):
                plan_text += f"- Expected state: {current_step.expected_state}\n"
            if upcoming:
                plan_text += "Upcoming steps (context only):\n" + "\n".join(upcoming[:4])
        elif plan:
            plan_lines = [f"- Step {s.id}: {s.description} (status={s.status})" for s in plan.steps]
            plan_text = "Plan:\n" + "\n".join(plan_lines)

        loop_state_text = ""
        if loop_state:
            notebook = loop_state.get("notebook_summary", "")
            if notebook:
                loop_state_text += f"\n{notebook}\n"
            
            loop_bits = [f"{k}={v}" for k, v in loop_state.items() if v not in (None, "") and k != "notebook_summary"]
            if loop_bits:
                loop_state_text += "Loop state: " + ", ".join(loop_bits)

        ax_context = ""
        som_context = ""
        if ax_tree:
            ax_str = self._summarize_ax_tree(ax_tree)
            ax_context = f"\nVisible UI Semantic Structure (summarized):\n{ax_str}\n"
        if som_tags:
            som_lines = []
            for tag in som_tags[:50]:
                frame = tag.get("frame", {})
                som_lines.append(
                    f"#{tag.get('id')}: role={tag.get('role','')} label={tag.get('label','')} "
                    f"frame=({frame.get('x','?')},{frame.get('y','?')},{frame.get('w','?')},{frame.get('h','?')}) (logical pts)"
                )
            som_context = (
                "\nNumbered overlay marks are drawn on the screenshot. "
                "Use element_id to reference these instead of guessing coordinates.\n"
                + "\n".join(som_lines)
            )

        system_prompt = f"""
            You are a cautious, focused macOS desktop operator. You have a robust toolbox including:
            - `computer`: for low-level mouse/keyboard interaction and UI inspection (`inspect_ui`).
            - `browser`: for high-speed reading and navigation of web pages (use this for research).
            - `notebook`: for saving facts and notes to persistent memory (use this to avoid forgetting things).
            - `shell`: for local workspace file operations.

            At each step you see a single screenshot of the current display plus a short textual history of previous actions and
            observations.
            - You may return a *macro action* by supplying `actions: [...]` to batch multiple low-level steps in one call.
            {plan_text}
            {loop_state_text}
            {ax_context}
            {som_context}

            Planning & Thinking
            - Always reason from what is currently visible: windows, icons, menus.
            - Use the provided Accessibility Tree and numbered overlay marks to ground actions. If a tag exists, return its ID via `element_id` instead of guessing coordinates.
            - Use `inspect_ui` if visual elements are ambiguous or you need to find hidden controls.
            - Coordinates: only provide x/y when no overlay tag is available. If using x/y, return logical display points (screenshot is already downscaled to logical resolution).
            - To reduce latency, prefer batching obvious sequences (e.g., click + type + enter) using `actions`.
            - For Research:
              1. Use `browser` tool to `get_links` or `get_page_content`.
              2. Read the content.
              3. SAVE key findings using `notebook` tool (`add_note`).
              4. This prevents data loss when context window fills up.

            Environment
            - System: {self.system_info}
            - Screenshot resolution (logical, downscaled): {self.display.logical_width}x{self.display.logical_height} pixels.
            - Display scale factor: {self.display.scale_factor} (HID will convert logical points to physical automatically).
            - (0, 0) is top-left.
            
            Safety
            - No destructive actions.
            - No network access via shell (use browser tool).
            - `shell` is sandboxed.
            
            Action Selection
            - Prefer batching obvious sequences using the `actions` array (macro_actions) to cut latency.
            - Prefer `browser` tools over `computer` OCR/Vision for text-heavy web tasks.
            - Prefer `inspect_ui` over random guessing of coordinates.
            
            Recent events:
            {history[-10:]}
        """
        if repeat_info and repeat_info.get("count", 0) >= 2:
            system_prompt += (
                f" Warning: last action repeated {repeat_info['count']} times "
                f"({repeat_info.get('action')}); choose a different next action."
            )
        if repeat_info and repeat_info.get("hint"):
            system_prompt += f" Hint from verifier: {repeat_info['hint']}."

        mime = "image/png" if self.settings.encode_format.lower() == "png" else "image/jpeg"

        task_hint = f"User request: {user_prompt}" if user_prompt else "No explicit user task provided."
        content = [
            {
                "type": "text",
                "text": f"{task_hint}\n\nPlan the next step. Prefer a single macro action (actions array) when multiple sequential steps are obvious.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{observation_b64}"},
            },
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Prepare reasoning parameters
        extra_body = {}
        if self.settings.reasoning_effort or self.settings.reasoning_max_tokens:
            reasoning_config = {}
            if self.settings.reasoning_effort:
                reasoning_config["effort"] = self.settings.reasoning_effort
            elif self.settings.reasoning_max_tokens:  # Use elif to ensure mutual exclusivity
                reasoning_config["max_tokens"] = self.settings.reasoning_max_tokens
            
            # Only add 'reasoning' to extra_body if at least one config is present
            if reasoning_config:
                extra_body["reasoning"] = reasoning_config

        return self.client.chat.completions.create(
            model=self.settings.openrouter_model,
            messages=messages,
            tools=[COMPUTER_TOOL, SHELL_TOOL, NOTEBOOK_TOOL, BROWSER_TOOL],
            tool_choice="auto",
            extra_body=extra_body if extra_body else None,
        )

    def _parse_tool_call(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract the first tool call and map it to the local action schema."""
        choices = getattr(response, "choices", [])
        if not choices:
            return None
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            # No tool call means the model replied with text; treat as noop.
            return {"type": "noop", "reason": "model returned text"}

        first = tool_calls[0]
        tool_name = getattr(first.function, "name", None)
        args_raw = first.function.arguments if hasattr(first, "function") else "{}"
        try:
            args = json.loads(args_raw or "{}")
        except json.JSONDecodeError:
            return {"type": "noop", "reason": f"bad tool args: {args_raw}"}

        if tool_name == "computer":
            return self._map_tool_args(args)
        if tool_name == "shell":
            return self._map_shell_args(args)
        if tool_name == "notebook":
            return self._map_notebook_args(args)
        if tool_name == "browser":
            return self._map_browser_args(args)
        return {"type": "noop", "reason": f"unknown tool {tool_name}"}

    def _map_tool_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Macro-action path: a list of sub-actions
        if isinstance(args.get("actions"), list):
            mapped_actions = []
            for sub in args["actions"]:
                if not isinstance(sub, dict):
                    continue
                mapped = self._map_single_computer_action(sub)
                if mapped.get("type") != "noop":
                    mapped_actions.append(mapped)
            if not mapped_actions:
                return {"type": "noop", "reason": "macro_actions provided but no valid sub-actions"}
            return {"type": "macro_actions", "actions": mapped_actions}

        return self._map_single_computer_action(args)

    def _map_single_computer_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action")
        verify_after = args.get("verify_after")
        
        def _apply_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
            if verify_after is not None:
                payload["verify_after"] = bool(verify_after)
            return payload
        
        # Common phantom_mode handling for click/hover actions
        phantom_mode = args.get("phantom_mode", False)
        
        if action == "move_mouse":
            x = args.get("x")
            y = args.get("y")
            payload = {"type": "mouse_move"}
            if args.get("element_id") is not None:
                payload["element_id"] = args.get("element_id")
            if x is None or y is None:
                if "element_id" in payload:
                    return _apply_verify(payload)
                return {"type": "noop", "reason": "move_mouse missing coordinates"}
            payload["x"] = float(x)
            payload["y"] = float(y)
            return _apply_verify(payload)

        if action in ("left_click", "right_click", "double_click"):
            payload = {"type": action}
            if args.get("element_id") is not None:
                payload["element_id"] = args.get("element_id")
            if args.get("x") is not None and args.get("y") is not None:
                payload["x"] = float(args.get("x"))
                payload["y"] = float(args.get("y"))
            if phantom_mode:
                payload["phantom_mode"] = True
            return _apply_verify(payload)

        if action == "drag_and_drop":
            payload = {"type": "drag_and_drop"}
            # Source
            if args.get("element_id") is not None:
                payload["element_id"] = args.get("element_id")
            if args.get("x") is not None and args.get("y") is not None:
                payload["x"] = float(args.get("x"))
                payload["y"] = float(args.get("y"))
            
            # Destination
            if args.get("target_x") is not None and args.get("target_y") is not None:
                payload["target_x"] = float(args.get("target_x"))
                payload["target_y"] = float(args.get("target_y"))
            else:
                return {"type": "noop", "reason": "drag_and_drop missing target coordinates"}
            
            payload["duration"] = float(args.get("duration", 0.5))
            payload["hold_delay"] = float(args.get("hold_delay", 0.0))
            return _apply_verify(payload)
        
        if action == "select_area":
            payload = {"type": "select_area"}
            if args.get("x") is not None and args.get("y") is not None:
                payload["x"] = float(args.get("x"))
                payload["y"] = float(args.get("y"))
            if args.get("target_x") is not None and args.get("target_y") is not None:
                payload["target_x"] = float(args.get("target_x"))
                payload["target_y"] = float(args.get("target_y"))
            else:
                return {"type": "noop", "reason": "select_area missing target coordinates"}
            payload["duration"] = float(args.get("duration", 0.4))
            payload["hold_delay"] = float(args.get("hold_delay", 0.0))
            return _apply_verify(payload)

        if action == "hover":
            payload = {"type": "hover"}
            if args.get("element_id") is not None:
                payload["element_id"] = args.get("element_id")
            if args.get("x") is not None and args.get("y") is not None:
                payload["x"] = float(args.get("x"))
                payload["y"] = float(args.get("y"))
            payload["duration"] = float(args.get("duration", 1.0))
            return _apply_verify(payload)

        if action == "probe_ui":
            payload = {"type": "probe_ui"}
            if args.get("x") is not None and args.get("y") is not None:
                payload["x"] = float(args.get("x"))
                payload["y"] = float(args.get("y"))
            if args.get("radius") is not None:
                payload["radius"] = float(args.get("radius"))
            return _apply_verify(payload)

        if action == "clipboard_op":
            sub = args.get("sub_action")
            if not sub:
                return {"type": "noop", "reason": "clipboard_op missing sub_action"}
            payload = {"type": "clipboard_op", "sub_action": sub}
            if sub == "write":
                payload["content"] = args.get("content", "")
            return _apply_verify(payload)

        if action == "scroll":
            return _apply_verify({
                "type": "scroll", 
                "clicks": int(args.get("scroll_y", 0)),
                "axis": args.get("axis", "vertical")
            })
        if action == "type":
            return _apply_verify({"type": "type", "text": args.get("text", "")})
        if action == "hotkey":
            return _apply_verify({"type": "key", "keys": args.get("keys") or []})
        if action == "wait":
            return _apply_verify({"type": "wait", "seconds": float(args.get("seconds", 1))})
        if action == "screenshot":
            return _apply_verify({"type": "capture_only", "reason": "model requested screenshot"})
        if action == "open_app":
            return _apply_verify({"type": "open_app", "app_name": args.get("app_name", "")})
        if action == "inspect_ui":
            return _apply_verify({"type": "inspect_ui"})
        if action == "run_skill":
            return _apply_verify({
                "type": "run_skill",
                "skill_id": args.get("skill_id"),
                "skill_name": args.get("skill_name"),
            })

        return {"type": "noop", "reason": f"unknown action {action}"}

    def _map_shell_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        command = args.get("command") or ""
        cwd = args.get("cwd")
        if not command:
            return {"type": "noop", "reason": "shell command missing"}

        return {
            "type": "sandbox_shell",
            "cmd": command,
            "cwd": cwd,
            "execution": "shell",
        }

    def _map_notebook_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action")
        return {
            "type": "notebook_op",
            "action": action,
            "content": args.get("content", ""),
            "source": args.get("source", "agent"),
            "execution": "notebook"
        }

    def _map_browser_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "browser_op",
            "command": args.get("command"),
            "app_name": args.get("app_name", "Safari"),
            "url": args.get("url"),
            "execution": "browser"
        }

    def _summarize_ax_tree(self, tree: Dict[str, Any], max_nodes: int = 80, max_depth: int = 4) -> str:
        """
        Produce a concise, depth-limited summary of the AX tree to cut token usage.
        Keeps only role/title/value/frame and limits node count.
        """
        lines: List[str] = []
        truncated = False
        interactive_roles = {"AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox", "AXComboBox", "AXMenuItem"}

        def _walk(node: Dict[str, Any], depth: int) -> None:
            nonlocal truncated
            if len(lines) >= max_nodes:
                truncated = True
                return

            role = (node.get("role") or "node").strip()
            title = (node.get("title") or "").strip()
            value = (node.get("value") or "").strip()
            frame = node.get("frame") or {}
            has_frame = frame and frame.get("w", 0) > 0 and frame.get("h", 0) > 0

            # Skip verbose containers with no grounding value
            if has_frame or title or value or role in interactive_roles:
                frame_str = (
                    f"({frame.get('x','?')},{frame.get('y','?')},{frame.get('w','?')},{frame.get('h','?')})"
                    if has_frame else "(no frame)"
                )
                lines.append(f"[d{depth}] role={role} title={title or '-'} value={value or '-'} frame={frame_str}")

            if depth >= max_depth:
                if node.get("children"):
                    truncated = True
                return

            for child in node.get("children") or []:
                if len(lines) >= max_nodes:
                    truncated = True
                    return
                _walk(child, depth + 1)

        _walk(tree, 0)
        summary = "\n".join(lines)
        if truncated:
            summary += "\n...[AX tree truncated]"
        return summary
