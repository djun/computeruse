from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DisplayInfo:
    logical_width: int
    logical_height: int
    physical_width: int
    physical_height: int
    scale_factor: float


ComputerActionName = Literal[
    "move_mouse",
    "left_click",
    "right_click",
    "double_click",
    "drag_and_drop",
    "select_area",
    "hover",
    "probe_ui",
    "inspect_ui",
    "scroll",
    "type",
    "hotkey",
    "wait",
    "wait_for_element",
    "wait_for_idle",
    "click_element",
    "fill_field",
    "scroll_to_element",
    "click_and_type",
    "clipboard_op",
    "read_clipboard",
    "write_clipboard",
    "open_app",
    "focus_window",
    "screenshot",
    "zoom",
    "observe",
    "run_skill",
    "done",
    "ask_user",
    # Clean-API aliases (mapped internally onto the legacy types above).
    "click",
    "input_text",
    "press_keys",
    "drag",
    "focus",
    "clipboard",
]


COMPUTER_ACTION_SPACE: tuple[ComputerActionName, ...] = (
    "move_mouse",
    "left_click",
    "right_click",
    "double_click",
    "drag_and_drop",
    "select_area",
    "hover",
    "probe_ui",
    "inspect_ui",
    "scroll",
    "type",
    "hotkey",
    "wait",
    "wait_for_element",
    "wait_for_idle",
    "click_element",
    "fill_field",
    "scroll_to_element",
    "click_and_type",
    "clipboard_op",
    "read_clipboard",
    "write_clipboard",
    "open_app",
    "focus_window",
    "screenshot",
    "zoom",
    "observe",
    "run_skill",
    "done",
    "ask_user",
    "click",
    "input_text",
    "press_keys",
    "drag",
    "focus",
    "clipboard",
)


@dataclass(frozen=True)
class Capability:
    """Availability of one runtime capability for the current OS/profile/flags.

    mode: real (works), dry_run (accepted but no real input is sent),
    degraded (works partially/unreliably), blocked (will fail; do not attempt).
    """

    name: str
    available: bool
    mode: str  # real | dry_run | degraded | blocked
    reason: str = ""


# Internal action types that observe state instead of mutating it. They end the
# reasoning turn: fresh context must reach the model before any further action,
# so they may not be mixed with executable sub-actions inside a macro.
OBSERVATION_ACTION_TYPES: frozenset[str] = frozenset(
    {"capture_only", "zoom", "inspect_ui", "probe_ui"}
)

# Loop-control action types resolved by the orchestrator without touching the
# computer: `done` ends the task, `noop` skips the turn, `invalid_action`
# feeds a mapping error back to the model, `ask_user` requests human input.
LOOP_CONTROL_ACTION_TYPES: frozenset[str] = frozenset({"done", "noop", "invalid_action", "ask_user"})
