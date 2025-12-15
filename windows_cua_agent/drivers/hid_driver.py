from __future__ import annotations

import ctypes
import os
import time
from typing import Iterable, Tuple

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.coordinates import clamp_point, point_to_px
from cua_agent.utils.logger import get_logger
from windows_cua_agent.utils.windows_integration import get_display_info


class HIDDriver:
    """Executes low-level HID events. Defaults to dry-run unless ENABLE_HID is true."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.enabled = bool(settings.enable_hid)
        self.display = get_display_info()

    def move(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self._mouse_move(x, y), f"move({x},{y})")

    def left_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self._click(x, y, button="left"), f"left_click({x},{y})")

    def right_click(self, x: float, y: float) -> ActionResult:
        return self._perform(lambda: self._click(x, y, button="right"), f"right_click({x},{y})")

    def double_click(self, x: float, y: float) -> ActionResult:
        def _action() -> None:
            self._click(x, y, button="left")
            time.sleep(0.05)
            self._click(x, y, button="left")

        return self._perform(_action, f"double_click({x},{y})")

    def type_text(self, text: str) -> ActionResult:
        return self._perform(lambda: self._type_text_unicode(text), f"type:{text!r}")

    def press_keys(self, keys: Iterable[str]) -> ActionResult:
        combo = [self._normalize_key(k) for k in keys if k]
        label = "hotkey:" + "+".join(combo)
        return self._perform(lambda: self._hotkey(combo), label)

    def scroll(self, clicks: int, axis: str = "vertical") -> ActionResult:
        return self._perform(lambda: self._scroll(clicks, axis=axis), f"scroll({axis}):{clicks}")

    def drag_and_drop(
        self,
        x: float,
        y: float,
        tx: float,
        ty: float,
        duration: float = 0.5,
        hold_delay: float = 0.0,
    ) -> ActionResult:
        def _action() -> None:
            self._mouse_move(x, y)
            if hold_delay > 0:
                time.sleep(float(hold_delay))
            self._mouse_button("left", down=True)
            self._smooth_move(tx, ty, duration=float(duration))
            self._mouse_button("left", down=False)

        return self._perform(_action, f"drag({x},{y}->{tx},{ty})")

    def select_area(
        self,
        x: float,
        y: float,
        tx: float,
        ty: float,
        duration: float = 0.4,
        hold_delay: float = 0.0,
    ) -> ActionResult:
        def _action() -> None:
            self._mouse_move(x, y)
            if hold_delay > 0:
                time.sleep(float(hold_delay))
            self._mouse_button("left", down=True)
            self._smooth_move(tx, ty, duration=float(duration))
            self._mouse_button("left", down=False)

        return self._perform(_action, f"select_area({x},{y}->{tx},{ty})")

    def hover(self, x: float, y: float, duration: float = 1.0) -> ActionResult:
        def _action() -> None:
            self._mouse_move(x, y)
            time.sleep(float(duration))

        return self._perform(_action, f"hover({x},{y}, {duration}s)")

    def _perform(self, func, label: str) -> ActionResult:
        if not self.enabled:
            self.logger.info("Dry-run HID: %s", label)
            return ActionResult(success=True, reason="dry-run")
        try:
            func()
            self.logger.info("HID action executed: %s", label)
            return ActionResult(success=True)
        except Exception as exc:
            self.logger.error("HID action failed: %s", exc)
            return ActionResult(success=False, reason=str(exc))

    def _to_px(self, x: float, y: float) -> Tuple[int, int]:
        x, y = clamp_point(float(x), float(y), self.display.logical_width, self.display.logical_height)
        return point_to_px(x, y, self.display.scale_factor)

    # --- Mouse (SendInput) ---

    def _mouse_move(self, x: float, y: float) -> None:
        px, py = self._to_px(x, y)
        self._send_mouse_move_abs(px, py)

    def _smooth_move(self, x: float, y: float, duration: float) -> None:
        if duration <= 0:
            return self._mouse_move(x, y)

        steps = max(4, min(60, int(duration / 0.01)))
        # Compute intermediate points in logical space for smoother movement.
        start_x, start_y = self._get_cursor_pos_logical()
        for i in range(1, steps + 1):
            t = i / steps
            ix = start_x + (x - start_x) * t
            iy = start_y + (y - start_y) * t
            self._mouse_move(ix, iy)
            time.sleep(duration / steps)

    def _get_cursor_pos_logical(self) -> tuple[float, float]:
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            pt = POINT()
            if user32.GetCursorPos(ctypes.byref(pt)):
                # Convert physical pixels -> logical
                return (pt.x / self.display.scale_factor, pt.y / self.display.scale_factor)
        except Exception:
            pass
        return (0.0, 0.0)

    def _click(self, x: float, y: float, button: str = "left") -> None:
        self._mouse_move(x, y)
        self._mouse_button(button, down=True)
        self._mouse_button(button, down=False)

    def _mouse_button(self, button: str, *, down: bool) -> None:
        button_l = button.lower()
        if button_l == "left":
            flag = 0x0002 if down else 0x0004  # LEFTDOWN/LEFTUP
        elif button_l == "right":
            flag = 0x0008 if down else 0x0010  # RIGHTDOWN/RIGHTUP
        else:
            raise ValueError(f"unsupported button: {button}")
        self._send_mouse_event(flag, dx=0, dy=0, mouse_data=0)

    def _scroll(self, clicks: int, axis: str = "vertical") -> None:
        # clicks are in "notches"; Windows expects multiples of WHEEL_DELTA (120).
        WHEEL_DELTA = 120
        axis_l = (axis or "vertical").lower()
        if axis_l == "horizontal":
            flag = 0x01000  # MOUSEEVENTF_HWHEEL
        else:
            flag = 0x0800  # MOUSEEVENTF_WHEEL
        self._send_mouse_event(flag, dx=0, dy=0, mouse_data=int(clicks) * WHEEL_DELTA)

    def _send_mouse_move_abs(self, x_px: int, y_px: int) -> None:
        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_ABSOLUTE = 0x8000
        dx, dy = self._normalize_absolute(x_px, y_px)
        self._send_mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, dx=dx, dy=dy, mouse_data=0)

    def _normalize_absolute(self, x_px: int, y_px: int) -> tuple[int, int]:
        w = max(1, int(self.display.physical_width) - 1)
        h = max(1, int(self.display.physical_height) - 1)
        x_px = max(0, min(int(x_px), w))
        y_px = max(0, min(int(y_px), h))
        dx = int(round(x_px * 65535 / w))
        dy = int(round(y_px * 65535 / h))
        return dx, dy

    # --- Keyboard (SendInput) ---

    def _hotkey(self, keys: list[str]) -> None:
        if not keys:
            return
        modifiers = [k for k in keys if k in {"ctrl", "shift", "alt", "win"}]
        main = [k for k in keys if k not in {"ctrl", "shift", "alt", "win"}]

        # Modifier-only combos (e.g., ["win"]) should behave like a tap.
        if not main:
            for k in modifiers:
                self._key_event(k, down=True)
                self._key_event(k, down=False)
            return

        for mod in modifiers:
            self._key_event(mod, down=True)
        for k in main:
            self._key_event(k, down=True)
            self._key_event(k, down=False)
        for mod in reversed(modifiers):
            self._key_event(mod, down=False)

    def _type_text_unicode(self, text: str) -> None:
        if not text:
            return
        # Send UTF-16 code units so surrogate pairs (emoji) work.
        data = text.encode("utf-16-le")
        for i in range(0, len(data), 2):
            code_unit = int.from_bytes(data[i : i + 2], "little")
            self._send_key_unicode(code_unit, down=True)
            self._send_key_unicode(code_unit, down=False)
            time.sleep(0.002)

    def _key_event(self, key: str, *, down: bool) -> None:
        vk = self._vk_code_for_key(key)
        self._send_key_vk(vk, down=down)

    def _vk_code_for_key(self, key: str) -> int:
        k = (key or "").lower()
        if len(k) == 1 and ("a" <= k <= "z" or "0" <= k <= "9"):
            return ord(k.upper())

        mapping = {
            "backspace": 0x08,
            "tab": 0x09,
            "enter": 0x0D,
            "return": 0x0D,
            "shift": 0x10,
            "ctrl": 0x11,
            "control": 0x11,
            "alt": 0x12,
            "option": 0x12,
            "pause": 0x13,
            "capslock": 0x14,
            "esc": 0x1B,
            "escape": 0x1B,
            "space": 0x20,
            "pageup": 0x21,
            "pagedown": 0x22,
            "end": 0x23,
            "home": 0x24,
            "left": 0x25,
            "up": 0x26,
            "right": 0x27,
            "down": 0x28,
            "insert": 0x2D,
            "delete": 0x2E,
            "win": 0x5B,  # VK_LWIN
        }
        if k in mapping:
            return mapping[k]

        if k.startswith("f") and k[1:].isdigit():
            n = int(k[1:])
            if 1 <= n <= 24:
                return 0x70 + (n - 1)

        raise ValueError(f"unsupported key: {key}")

    def _send_key_vk(self, vk: int, *, down: bool) -> None:
        KEYEVENTF_KEYUP = 0x0002
        flags = 0 if down else KEYEVENTF_KEYUP
        self._send_keyboard_event(vk=vk, scan=0, flags=flags)

    def _send_key_unicode(self, code_unit: int, *, down: bool) -> None:
        KEYEVENTF_UNICODE = 0x0004
        KEYEVENTF_KEYUP = 0x0002
        flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if not down else 0)
        self._send_keyboard_event(vk=0, scan=code_unit, flags=flags)

    # --- SendInput bindings ---

    def _send_mouse_event(self, flags: int, *, dx: int, dy: int, mouse_data: int) -> None:
        inputs = [self._make_mouse_input(dx=dx, dy=dy, mouse_data=mouse_data, flags=flags)]
        self._send_inputs(inputs)

    def _send_keyboard_event(self, *, vk: int, scan: int, flags: int) -> None:
        inputs = [self._make_keyboard_input(vk=vk, scan=scan, flags=flags)]
        self._send_inputs(inputs)

    def _send_inputs(self, inputs: list[ctypes.Structure]) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        n = len(inputs)
        if n <= 0:
            return
        arr_type = INPUT * n
        arr = arr_type(*inputs)  # type: ignore[arg-type]
        sent = int(user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT)))
        if sent != n:
            raise OSError(f"SendInput sent {sent}/{n}")

    def _make_mouse_input(self, *, dx: int, dy: int, mouse_data: int, flags: int):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=mouse_data, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def _make_keyboard_input(self, *, vk: int, scan: int, flags: int):
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def _normalize_key(self, key: str) -> str:
        mapping = {
            "cmd": "win",
            "command": "win",
            "control": "ctrl",
            "option": "alt",
            "return": "enter",
            "escape": "esc",
            "windows": "win",
        }
        lower = (key or "").lower()
        return mapping.get(lower, lower)


# --- ctypes structures/constants (Win32 INPUT) ---

ULONG_PTR = ctypes.c_ulonglong if os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]
