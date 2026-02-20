from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from cua_agent.agent.state_manager import VERIFICATION_SENSOR_HIERARCHY
from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cua_agent.orchestrator.planning import Plan, Step


@dataclass
class ToolRegistration:
    """Tool schema + routing contract used by CognitiveCore."""

    name: str
    schema: Dict[str, Any]
    enabled: Callable[["CognitiveCore"], bool]
    mapper: Callable[["CognitiveCore", Dict[str, Any]], Dict[str, Any]]


# OpenRouter exposes an OpenAI-compatible tool-calling API. We define our own
# computer tool schema so Claude Opus 4.5 can drive local actions.
COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Control the desktop: move and click the mouse, type, press hotkeys, "
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
                            "verification": {
                                "type": "object",
                                "description": (
                                    "Post-action verification contract. "
                                    "Prefer low-cost sensors before vision."
                                ),
                                "properties": {
                                    "sensor": {
                                        "type": "string",
                                        "enum": ["none", "os_telemetry", "a11y_tree", "pixel_diff", "vision_full"],
                                    },
                                    "expected_state": {"type": "string"},
                                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 30, "default": 5},
                                },
                                "required": ["sensor"],
                                "additionalProperties": False,
                            },
                            "skill_id": {"type": "string"},
                            "skill_name": {"type": "string"},
                            "skill_args": {
                                "type": "object",
                                "description": "Runtime arguments for parameterized skills.",
                                "additionalProperties": {
                                    "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]
                                },
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Short reason for this action choice (for debug observability).",
                            },
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
                "verification": {
                    "type": "object",
                    "description": (
                        "Verification contract for this action block. "
                        "Always choose the cheapest sensor that can prove success."
                    ),
                    "properties": {
                        "sensor": {
                            "type": "string",
                            "enum": ["none", "os_telemetry", "a11y_tree", "pixel_diff", "vision_full"],
                        },
                        "expected_state": {
                            "type": "string",
                            "description": (
                                "Concrete expected state, e.g. "
                                "'text_exists:Dashboard', 'clipboard_changed', or 'file_exists:C:/tmp/out.pdf'."
                            ),
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "default": 5,
                        },
                    },
                    "required": ["sensor"],
                    "additionalProperties": False,
                },
                "skill_id": {
                    "type": "string",
                    "description": "ID of a stored procedural skill to execute (run_skill).",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of a stored procedural skill to execute (run_skill).",
                },
                "skill_args": {
                    "type": "object",
                    "description": "Runtime argument values for a parameterized stored skill.",
                    "additionalProperties": {
                        "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]
                    },
                },
                "rationale": {
                    "type": "string",
                    "description": "Short reason for choosing this action (for debug observability).",
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

SCRIPT_TOOL = {
    "type": "function",
    "function": {
        "name": "script",
        "description": (
            "Safer script workflow inside the sandbox workspace. "
            "Write/update scripts, read files, and run approved script files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["write", "read", "run"]},
                "path": {
                    "type": "string",
                    "description": "Relative path under the sandbox workspace (e.g., tools/report.py).",
                },
                "content": {
                    "type": "string",
                    "description": "Script content used by action=write.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "When false, write fails if file already exists.",
                    "default": True,
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional command-line arguments used by action=run.",
                },
                "runtime_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120,
                    "description": "Optional runtime limit for action=run (capped by runtime policy).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional relative working directory under the agent workspace.",
                },
            },
            "required": ["action", "path"],
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
        "description": (
            "Interact with web browsers (Safari/Chrome). "
            "On macOS this is AppleScript/JXA-based; on Windows it requires a Chromium browser launched with remote debugging (CDP)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "get_page_content",
                        "get_links",
                        "navigate",
                        "fill_form",
                        "click_element",
                        "get_dom_tree",
                        "run_javascript",
                        "go_back",
                        "go_forward",
                        "reload"
                    ]
                },
                "app_name": {
                    "type": "string",
                    "description": "Safari or Google Chrome",
                    "default": "Safari"
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (for 'navigate' command)"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for targeting elements. Required for fill_form and click_element. (e.g., '#search-box', 'input[name=\"q\"]')"
                },
                "value": {
                    "type": "string",
                    "description": "Content to type for fill_form, or raw JavaScript code for run_javascript."
                }
            },
            "required": ["command"]
        }
    }
}

READ_ONLY_BROWSER_COMMANDS = {"get_page_content", "get_links", "get_dom_tree"}


class CognitiveCore:
    """Calls Claude Opus 4.5 via OpenRouter with a custom computer tool."""

    def __init__(self, settings: Settings, computer: ComputerAdapter) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.computer = computer
        self.display = computer.display
        self.system_info = computer.system_info
        self.platform_name = computer.platform_name
        self._tool_registry: Dict[str, ToolRegistration] = {}
        self._tool_order: List[str] = []
        self._register_default_tools()
        self.client = self._build_client()
        self._log_execution_profile_startup()

    def _tool_enabled_map(self) -> Dict[str, bool]:
        self._ensure_tool_registry()
        status: Dict[str, bool] = {}
        for tool_name in self._tool_order:
            registration = self._tool_registry.get(tool_name)
            if not registration:
                continue
            try:
                status[tool_name] = bool(registration.enabled(self))
            except Exception:
                status[tool_name] = False
        return status

    def _shell_tool_enabled(self) -> bool:
        return self.settings.allows_shell_actions() and bool(self.settings.enable_shell)

    def register_tool(self, registration: ToolRegistration, *, position: Optional[int] = None) -> None:
        """Register a tool schema and mapper for dynamic tool expansion."""
        tool_name = str(registration.name or "").strip()
        if not tool_name:
            raise ValueError("tool registration requires a non-empty name")
        schema_name = str(registration.schema.get("function", {}).get("name", "")).strip()
        if schema_name and schema_name != tool_name:
            raise ValueError(
                f"tool registration name mismatch: registration={tool_name} schema={schema_name}"
            )

        if tool_name in self._tool_registry:
            # Replace existing definition while preserving original order.
            self._tool_registry[tool_name] = registration
            return

        self._tool_registry[tool_name] = registration
        if position is None or position >= len(self._tool_order):
            self._tool_order.append(tool_name)
        else:
            self._tool_order.insert(max(position, 0), tool_name)

    def _register_default_tools(self) -> None:
        self.register_tool(
            ToolRegistration(
                name="notebook",
                schema=NOTEBOOK_TOOL,
                enabled=lambda core: True,
                mapper=lambda core, args: core._map_notebook_args(args),
            )
        )
        self.register_tool(
            ToolRegistration(
                name="computer",
                schema=COMPUTER_TOOL,
                enabled=lambda core: core.settings.allows_gui_actions(),
                mapper=lambda core, args: core._map_tool_args(args),
            )
        )
        self.register_tool(
            ToolRegistration(
                name="shell",
                schema=SHELL_TOOL,
                enabled=lambda core: core._shell_tool_enabled(),
                mapper=lambda core, args: core._map_shell_args(args),
            )
        )
        self.register_tool(
            ToolRegistration(
                name="script",
                schema=SCRIPT_TOOL,
                enabled=lambda core: core._shell_tool_enabled(),
                mapper=lambda core, args: core._map_script_args(args),
            )
        )
        self.register_tool(
            ToolRegistration(
                name="browser",
                schema=BROWSER_TOOL,
                enabled=lambda core: core.settings.allows_browser_actions(),
                mapper=lambda core, args: core._map_browser_args(args),
            )
        )

    def _ensure_tool_registry(self) -> None:
        # Some unit tests instantiate via __new__ and bypass __init__.
        if not hasattr(self, "_tool_registry"):
            self._tool_registry = {}
        if not hasattr(self, "_tool_order"):
            self._tool_order = []
        if not self._tool_registry:
            self._tool_order = []
            self._register_default_tools()

    def _log_execution_profile_startup(self) -> None:
        status = self._tool_enabled_map()
        enabled = [name for name, allowed in status.items() if allowed]
        disabled = [name for name, allowed in status.items() if not allowed]
        self.logger.info(
            "Execution profile '%s' active. Tools enabled: [%s]. Tools disabled: [%s].",
            self.settings.execution_profile,
            ", ".join(enabled) if enabled else "none",
            ", ".join(disabled) if disabled else "none",
        )

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
        include_visual_context: bool = True,
        user_prompt: Optional[str] = None,
        repeat_info: Optional[Dict[str, Any]] = None,
        plan: Optional["Plan"] = None,
        current_step: Optional["Step"] = None,
        loop_state: Optional[Dict[str, Any]] = None,
        ax_tree: Optional[Dict[str, Any]] = None,
        som_tags: Optional[List[Dict[str, Any]]] = None,
        relevant_skills: Optional[List[Any]] = None,
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
                include_visual_context=include_visual_context,
                user_prompt=user_prompt,
                repeat_info=repeat_info,
                plan=plan,
                current_step=current_step,
                loop_state=loop_state,
                ax_tree=ax_tree,
                som_tags=som_tags,
                relevant_skills=relevant_skills,
            )
            parsed_action = self._parse_tool_call(response)
            if parsed_action:
                model_text = self._extract_response_text(response)
                return self._annotate_with_debug_trace(
                    parsed_action,
                    model_text=model_text,
                    current_step=current_step,
                    loop_state=loop_state,
                    repeat_info=repeat_info,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("OpenRouter call failed; falling back to noop.", exc_info=exc)

        return {"type": "noop", "reason": "Failed to generate action"}

    def _call_openrouter(
        self,
        observation_b64: str,
        history: List[str],
        include_visual_context: bool,
        user_prompt: Optional[str],
        repeat_info: Optional[Dict[str, Any]],
        plan: Optional["Plan"],
        current_step: Optional["Step"],
        loop_state: Optional[Dict[str, Any]],
        ax_tree: Optional[Dict[str, Any]],
        som_tags: Optional[List[Dict[str, Any]]],
        relevant_skills: Optional[List[Any]],
    ) -> Any:
        """Send a vision + tool-calling request to OpenRouter."""
        windows_cyborg = self.platform_name.lower().startswith("windows") and self.settings.windows_cyborg_mode

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

        skills_context = ""
        if relevant_skills:
            skills_lines = []
            for s in relevant_skills:
                param_blob = ""
                if getattr(s, "parameters", None):
                    rendered_params = []
                    for key, spec in (s.parameters or {}).items():
                        required = False
                        description = ""
                        if isinstance(spec, dict):
                            required = bool(spec.get("required", False))
                            description = str(spec.get("description", "")).strip()
                        elif isinstance(spec, str):
                            description = spec.strip()
                        suffix = " (required)" if required else ""
                        if description:
                            rendered_params.append(f"{key}{suffix}: {description}")
                        else:
                            rendered_params.append(f"{key}{suffix}")
                    if rendered_params:
                        param_blob = " | params: " + "; ".join(rendered_params[:5])
                skills_lines.append(f"- {s.name} (ID: {s.id}): {s.description}{param_blob}")
            skills_context = (
                "\nRelevant Skills/Macros:\n"
                + "\n".join(skills_lines)
                + "\nUse `run_skill` with the ID and pass `skill_args` when the skill exposes params.\n"
            )

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

        tool_status = self._tool_enabled_map()
        allow_gui = bool(tool_status.get("computer", False))
        allow_browser = bool(tool_status.get("browser", False))
        allow_shell = bool(tool_status.get("shell", False))
        allow_script = bool(tool_status.get("script", False))
        allow_notebook = bool(tool_status.get("notebook", False))

        tool_lines = []
        if allow_gui:
            tool_lines.append("- `computer`: for low-level mouse/keyboard interaction and UI inspection (`inspect_ui`).")
        if allow_browser:
            tool_lines.append(
                "- `browser`: CDP-based on Windows and may be unavailable; prefer `computer` for web automation."
                if windows_cyborg
                else "- `browser`: for high-speed reading/navigation of web pages."
            )
        if allow_shell:
            tool_lines.append("- `shell`: for local workspace file operations in the sandbox.")
        if allow_script:
            tool_lines.append("- `script`: safer write/read/run flow for workspace scripts with stricter guardrails.")
        if allow_notebook:
            tool_lines.append("- `notebook`: for saving facts and notes to persistent memory (use this to avoid forgetting things).")

        if allow_browser:
            browser_research_lines = (
                "- For Research:\n"
                "  1. Use `browser` tool to `get_links` or `get_page_content`.\n"
                "  2. Read the content.\n"
                "  3. SAVE key findings using `notebook` tool (`add_note`).\n"
                "  4. This prevents data loss when context window fills up."
                if not windows_cyborg
                else "- For Research on Windows Cyborg mode:\n"
                "  1. Prefer `inspect_ui` + the accessibility summary + screenshot grounding.\n"
                "  2. If you must use `browser`, it may fail unless CDP is enabled; switch back to `computer` immediately on CDP errors.\n"
                "  3. SAVE key findings using `notebook` (`add_note`)."
            )
            browser_preference_line = (
                "- Prefer `browser` tools over `computer` OCR/Vision for text-heavy web tasks."
                if not windows_cyborg
                else "- Windows Cyborg mode: prefer `computer` + `inspect_ui` + HID/Phantom Mode; avoid `browser` unless CDP is confirmed working."
            )
        elif allow_gui:
            browser_research_lines = (
                "- Browser tool is disabled in this execution profile; use `inspect_ui`, semantic grounding, and notebook notes."
            )
            browser_preference_line = "- Browser tool unavailable in current execution profile."
        else:
            browser_research_lines = (
                "- GUI/browser tools are disabled in this execution profile; solve tasks through `shell`, `script`, and `notebook`."
                if (allow_shell or allow_script)
                else "- GUI/browser tools are disabled and `shell` is unavailable; collect notes with `notebook` and request a profile/config change when execution is required."
            )
            browser_preference_line = "- GUI/browser actions are disabled in current execution profile."

        shell_safety_lines = (
            "- No network access via shell/script (use browser tool when available).\n- `shell` and `script` run inside a sandboxed workspace."
            if (allow_shell or allow_script)
            else "- `shell`/`script` tools are disabled (execution profile and/or ENABLE_SHELL=false)."
        )
        visual_context_line = (
            "- This turn includes an up-to-date screenshot."
            if include_visual_context
            else "- This turn has no screenshot. Re-plan from history + semantic state only; if uncertain, request `vision_full` verification."
        )

        system_prompt = f"""
            You are a high-efficiency {self.platform_name} autonomous desktop operator.
            Execution profile: {self.settings.execution_profile}.
            Toolbox:
            {chr(10).join(tool_lines)}

            At each step you receive textual history, and may also receive a screenshot of the current display.
            - You may return a *macro action* by supplying `actions: [...]` to batch multiple low-level steps in one call.
            - Every action block MUST include a `verification` contract with `sensor`, optional `expected_state`, and `timeout_seconds`.
            {plan_text}
            {loop_state_text}
            {skills_context}
            {ax_context}
            {som_context}

            Planning & Thinking
            {visual_context_line}
            - Always reason from what is currently visible: windows, icons, menus.
            - Use the provided Accessibility Tree and numbered overlay marks to ground actions. If a tag exists, return its ID via `element_id` instead of guessing coordinates.
            - Use `inspect_ui` if visual elements are ambiguous or you need to find hidden controls.
            - Coordinates: only provide x/y when no overlay tag is available. If using x/y, return logical display points (screenshot is already downscaled to logical resolution).
            - To reduce latency, prefer batching obvious sequences (e.g., click + type + enter) using `actions`.
            - Sensor pyramid (prefer cheapest): `none` -> `os_telemetry` -> `a11y_tree` -> `pixel_diff` -> `vision_full`.
            - Use `vision_full` only when context is lost, UI is purely visual, or cheaper validation failed.
            {browser_research_lines}

            Environment
            - System: {self.system_info}
            - Screenshot resolution (logical, downscaled): {self.display.logical_width}x{self.display.logical_height} pixels.
            - Display scale factor: {self.display.scale_factor} (HID will convert logical points to physical automatically).
            - (0, 0) is top-left.
            
            Safety
            - No destructive actions.
            {shell_safety_lines}
            
            Action Selection
            - Prefer batching obvious sequences using the `actions` array (macro_actions) to cut latency.
            {browser_preference_line}
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
        prompt_suffix = (
            "Plan the next step. Prefer a single macro action (actions array) when multiple sequential steps are obvious. "
            "Always return a verification contract for the action block."
        )
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"{task_hint}\n\n{prompt_suffix}",
            }
        ]
        if include_visual_context and observation_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{observation_b64}"},
                }
            )

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
            tools=self._available_tools(),
            tool_choice="auto",
            extra_body=extra_body if extra_body else None,
        )

    def _available_tools(self) -> List[Dict[str, Any]]:
        self._ensure_tool_registry()
        tools: List[Dict[str, Any]] = []
        for tool_name in self._tool_order:
            registration = self._tool_registry.get(tool_name)
            if not registration:
                continue
            try:
                if registration.enabled(self):
                    tools.append(registration.schema)
            except Exception as exc:
                self.logger.warning("tool '%s' enablement check failed: %s", tool_name, exc)
        return tools

    def _parse_tool_call(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract the first tool call and map it to the local action schema."""
        self._ensure_tool_registry()
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

        registration = self._tool_registry.get(str(tool_name or ""))
        if not registration:
            return {"type": "noop", "reason": f"unknown tool {tool_name}"}

        try:
            mapped = registration.mapper(self, args)
        except Exception as exc:
            self.logger.exception("tool mapper failed for '%s'", tool_name, exc_info=exc)
            return {"type": "noop", "reason": f"tool mapper failed: {tool_name}"}

        if isinstance(mapped, dict):
            return mapped
        return {"type": "noop", "reason": f"tool mapper returned invalid payload: {tool_name}"}

    def _map_tool_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.allows_gui_actions():
            return {
                "type": "noop",
                "reason": f"execution profile '{self.settings.execution_profile}' blocks GUI actions",
            }

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
            payload = {"type": "macro_actions", "actions": mapped_actions}
            contract = self._normalize_verification_contract(
                args.get("verification"),
                fallback_sensor="a11y_tree",
                verify_after=args.get("verify_after"),
            )
            if contract:
                payload["verification"] = contract
            if args.get("rationale"):
                payload["_debug_rationale"] = str(args.get("rationale"))[:600]
            return payload

        return self._map_single_computer_action(args)

    def _map_single_computer_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action")
        verify_after = args.get("verify_after")
        
        def _apply_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
            if verify_after is not None:
                payload["verify_after"] = bool(verify_after)
            contract = self._normalize_verification_contract(
                args.get("verification"),
                fallback_sensor=self._default_sensor_for_action_type(str(action or "")),
                verify_after=verify_after,
            )
            if contract:
                payload["verification"] = contract
            rationale = args.get("rationale")
            if rationale:
                payload["_debug_rationale"] = str(rationale)[:600]
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
            payload = {"type": "type", "text": args.get("text", "")}
            if args.get("element_id") is not None:
                payload["element_id"] = args.get("element_id")
            if args.get("phantom_mode") is not None:
                payload["phantom_mode"] = bool(args.get("phantom_mode"))
            # If the model provided an element_id but no explicit phantom flag, default to phantom for reliability.
            if args.get("element_id") is not None and args.get("phantom_mode") is None:
                payload["phantom_mode"] = True
            return _apply_verify(payload)
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
            payload = {
                "type": "run_skill",
                "skill_id": args.get("skill_id"),
                "skill_name": args.get("skill_name"),
            }
            raw_skill_args = args.get("skill_args")
            if isinstance(raw_skill_args, dict):
                cleaned_args: Dict[str, Any] = {}
                for key, value in raw_skill_args.items():
                    if isinstance(value, (str, int, float, bool)) and str(key).strip():
                        cleaned_args[str(key)] = value
                if cleaned_args:
                    payload["skill_args"] = cleaned_args
            return _apply_verify(payload)

        return {"type": "noop", "reason": f"unknown action {action}"}

    def _default_sensor_for_action_type(self, action_type: str) -> str:
        token = str(action_type or "").strip().lower()
        if token in {"wait", "screenshot", "inspect_ui", "probe_ui"}:
            return "none"
        if token in {"clipboard_op", "open_app"}:
            return "os_telemetry"
        if token in {"move_mouse", "hover", "scroll"}:
            return "none"
        if token in {"run_skill", "drag_and_drop", "select_area", "left_click", "right_click", "double_click", "hotkey", "key", "type"}:
            return "a11y_tree"
        return "pixel_diff"

    def _normalize_verification_contract(
        self,
        raw_contract: Any,
        *,
        fallback_sensor: str,
        verify_after: Any,
    ) -> Optional[Dict[str, Any]]:
        if verify_after is False and not isinstance(raw_contract, dict):
            return {"sensor": "none", "timeout_seconds": 1}

        if raw_contract is None and verify_after is None:
            return None

        contract = raw_contract if isinstance(raw_contract, dict) else {}

        fallback = str(fallback_sensor or "a11y_tree").strip().lower()
        if fallback not in VERIFICATION_SENSOR_HIERARCHY:
            fallback = "a11y_tree"

        sensor = str(contract.get("sensor") or fallback).strip().lower()
        if verify_after is False:
            sensor = "none"
        if sensor not in VERIFICATION_SENSOR_HIERARCHY:
            sensor = fallback

        timeout_raw = contract.get("timeout_seconds", 5)
        try:
            timeout_seconds = int(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = 5
        timeout_seconds = max(1, min(timeout_seconds, 30))
        if sensor == "none":
            timeout_seconds = 1

        expected_state_raw = contract.get("expected_state")
        expected_state = str(expected_state_raw).strip() if expected_state_raw is not None else None
        if expected_state == "":
            expected_state = None
        if expected_state and len(expected_state) > 500:
            expected_state = expected_state[:500]

        payload: Dict[str, Any] = {
            "sensor": sensor,
            "timeout_seconds": timeout_seconds,
        }
        if expected_state:
            payload["expected_state"] = expected_state
        return payload

    def _map_shell_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.allows_shell_actions():
            return {
                "type": "noop",
                "reason": f"execution profile '{self.settings.execution_profile}' blocks shell actions",
            }
        if not self.settings.enable_shell:
            return {
                "type": "noop",
                "reason": "shell disabled by ENABLE_SHELL=false",
            }

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

    def _map_script_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.settings.allows_shell_actions():
            return {
                "type": "noop",
                "reason": f"execution profile '{self.settings.execution_profile}' blocks script actions",
            }
        if not self.settings.enable_shell:
            return {
                "type": "noop",
                "reason": "script disabled by ENABLE_SHELL=false",
            }

        operation = str(args.get("action") or "").strip().lower()
        path = str(args.get("path") or "").strip()
        cwd = args.get("cwd")

        if operation not in {"write", "read", "run"}:
            return {"type": "noop", "reason": f"unknown script action {operation or 'none'}"}
        if not path:
            return {"type": "noop", "reason": "script path missing"}

        payload: Dict[str, Any] = {
            "type": "script_op",
            "operation": operation,
            "path": path,
            "cwd": cwd,
            "execution": "shell",
        }

        if operation == "write":
            payload["content"] = str(args.get("content") or "")
            payload["overwrite"] = bool(args.get("overwrite", True))
            return payload

        if operation == "run":
            raw_args = args.get("args")
            if isinstance(raw_args, list):
                cleaned = [str(item) for item in raw_args if item is not None]
                if cleaned:
                    payload["args"] = cleaned

            runtime_seconds = args.get("runtime_seconds")
            if runtime_seconds is not None:
                try:
                    payload["runtime_seconds"] = int(runtime_seconds)
                except (TypeError, ValueError):
                    return {"type": "noop", "reason": "script runtime_seconds must be an integer"}
            return payload

        return payload

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
        if not self.settings.allows_browser_actions():
            return {
                "type": "noop",
                "reason": f"execution profile '{self.settings.execution_profile}' blocks browser actions",
            }

        cmd = args.get("command")
        windows_cyborg = self.platform_name.lower().startswith("windows") and self.settings.windows_cyborg_mode
        if windows_cyborg:
            if cmd == "navigate":
                url = (args.get("url") or "").strip()
                if not url:
                    return {"type": "capture_only", "reason": "browser.navigate missing url (Windows Cyborg mode)"}
                payload = {
                    "type": "macro_actions",
                    "actions": [
                        {"type": "key", "keys": ["ctrl", "l"]},
                        {"type": "wait", "seconds": 0.15},
                        {"type": "type", "text": url},
                        {"type": "key", "keys": ["enter"]},
                    ],
                }
                payload["verification"] = self._normalize_verification_contract(
                    args.get("verification"),
                    fallback_sensor="a11y_tree",
                    verify_after=args.get("verify_after"),
                ) or {
                    "sensor": "a11y_tree",
                    "expected_state": self._default_cyborg_navigate_expected_state(url),
                    "timeout_seconds": 8,
                }
                return payload
            if cmd == "go_back":
                payload = {"type": "key", "keys": ["alt", "left"]}
                payload["verification"] = self._normalize_verification_contract(
                    args.get("verification"),
                    fallback_sensor="pixel_diff",
                    verify_after=args.get("verify_after"),
                ) or {"sensor": "pixel_diff", "timeout_seconds": 4}
                return payload
            if cmd == "go_forward":
                payload = {"type": "key", "keys": ["alt", "right"]}
                payload["verification"] = self._normalize_verification_contract(
                    args.get("verification"),
                    fallback_sensor="pixel_diff",
                    verify_after=args.get("verify_after"),
                ) or {"sensor": "pixel_diff", "timeout_seconds": 4}
                return payload
            if cmd == "reload":
                payload = {"type": "key", "keys": ["ctrl", "r"]}
                payload["verification"] = self._normalize_verification_contract(
                    args.get("verification"),
                    fallback_sensor="pixel_diff",
                    verify_after=args.get("verify_after"),
                ) or {"sensor": "pixel_diff", "timeout_seconds": 4}
                return payload

        payload = {
            "type": "browser_op",
            "command": cmd,
            "app_name": args.get("app_name", "Safari"),
            "url": args.get("url"),
            "selector": args.get("selector"),
            "value": args.get("value"),
            "execution": "browser"
        }
        default_contract = self._default_browser_verification_for_command(cmd)
        payload["verification"] = self._normalize_verification_contract(
            args.get("verification"),
            fallback_sensor=str(default_contract.get("sensor") or "pixel_diff"),
            verify_after=args.get("verify_after"),
        ) or default_contract
        return payload

    def _default_browser_verification_for_command(self, command: Any) -> Dict[str, Any]:
        token = str(command or "").strip().lower()
        if token in READ_ONLY_BROWSER_COMMANDS:
            return {"sensor": "none", "timeout_seconds": 1}
        return {"sensor": "pixel_diff", "timeout_seconds": 6}

    def _default_cyborg_navigate_expected_state(self, url: str) -> str:
        host = self._extract_url_host_token(url)
        if host:
            return f"url_contains:{host}"
        return "state_change"

    def _extract_url_host_token(self, url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""

        candidate = raw if "://" in raw else f"https://{raw}"
        host = ""
        try:
            parsed = urlparse(candidate)
            host = str(parsed.hostname or "").strip().lower()
        except Exception:
            host = ""

        if not host:
            token = raw.lower()
            for prefix in ("http://", "https://"):
                if token.startswith(prefix):
                    token = token[len(prefix):]
                    break
            host = token.split("/", 1)[0].split("?", 1)[0].strip().lower()

        if host.startswith("www.") and len(host) > 4:
            host = host[4:]
        return host[:160]

    def _extract_response_text(self, response: Any) -> str:
        """Extract plain assistant text from a tool-call response for debug telemetry."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks).strip()
        return str(content).strip()

    def _annotate_with_debug_trace(
        self,
        action: Dict[str, Any],
        model_text: str,
        current_step: Optional["Step"],
        loop_state: Optional[Dict[str, Any]],
        repeat_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attach concise cognitive trace metadata for the live dashboard."""
        trace_parts: List[str] = []
        if current_step:
            trace_parts.append(f"step {current_step.id}: {current_step.description}")
        if loop_state:
            failure_count = loop_state.get("failure_count")
            repeats = loop_state.get("repeat_without_change")
            if failure_count is not None or repeats is not None:
                trace_parts.append(f"loop failures={failure_count} repeat_without_change={repeats}")
        if repeat_info and repeat_info.get("hint"):
            trace_parts.append(f"hint: {repeat_info.get('hint')}")

        debug_rationale = action.get("_debug_rationale")
        if debug_rationale:
            trace_parts.append(f"rationale: {debug_rationale}")
        elif model_text:
            trace_parts.append(f"model_note: {model_text}")

        verification = action.get("verification")
        if isinstance(verification, dict):
            sensor = verification.get("sensor")
            expected = verification.get("expected_state")
            timeout = verification.get("timeout_seconds")
            trace_parts.append(
                f"verification: sensor={sensor} expected={expected or '-'} timeout={timeout or '-'}"
            )

        action_type = action.get("type", "unknown")
        trace_parts.append(f"selected_action: {action_type}")
        action["_debug_trace"] = "\n".join([part for part in trace_parts if part])[:1400]
        return action

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
