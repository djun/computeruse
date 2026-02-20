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
    "run_skill",
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
    "run_skill",
)
