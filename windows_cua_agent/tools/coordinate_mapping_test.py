from __future__ import annotations

import argparse
import sys
import time

from cua_agent.utils.config import Settings
from windows_cua_agent.drivers.hid_driver import HIDDriver
from windows_cua_agent.utils.windows_integration import get_display_info


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coordinate_mapping_test",
        description="Moves the mouse to known logical points and reports the observed cursor position.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually move the mouse. Requires ENABLE_HID=true.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=2.0,
        help="Countdown before movement begins (seconds).",
    )
    return parser.parse_args(argv)


def _get_cursor_logical() -> tuple[float, float]:
    # Local helper so this script doesn't depend on HIDDriver internals.
    import ctypes

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetCursorPos(ctypes.byref(pt))

    display = get_display_info()
    scale = float(display.scale_factor or 1.0)
    return float(pt.x) / scale, float(pt.y) / scale


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings()
    armed = bool(args.confirm) and bool(settings.enable_hid)

    display = get_display_info()
    print(
        "Display:",
        f"logical={display.logical_width}x{display.logical_height}",
        f"physical={display.physical_width}x{display.physical_height}",
        f"scale={display.scale_factor}",
    )

    if not settings.enable_hid:
        print("ENABLE_HID is false; driver will run in dry-run mode.")
        print("Set ENABLE_HID=true and re-run with --confirm to execute.")

    if args.confirm and not settings.enable_hid:
        print("Refusing to run: --confirm requires ENABLE_HID=true.")
        return 2

    # Force dry-run unless explicitly armed.
    settings.enable_hid = armed
    hid = HIDDriver(settings)

    start = _get_cursor_logical()
    print(f"Start cursor (logical): x={start[0]:.1f} y={start[1]:.1f}")

    points = [
        (display.logical_width * 0.5, display.logical_height * 0.5, "center"),
        (2.0, 2.0, "top-left"),
        (display.logical_width - 3.0, 2.0, "top-right"),
        (display.logical_width - 3.0, display.logical_height - 3.0, "bottom-right"),
        (2.0, display.logical_height - 3.0, "bottom-left"),
    ]

    if not armed:
        print("Mode: dry-run (no movement will occur).")
        return 0

    if args.delay_s > 0:
        print(f"Starting in {args.delay_s:.1f}s...")
        time.sleep(float(args.delay_s))

    for x, y, label in points:
        hid.move(float(x), float(y))
        time.sleep(0.15)
        observed = _get_cursor_logical()
        dx = observed[0] - float(x)
        dy = observed[1] - float(y)
        print(f"{label:>12}: target=({x:.1f},{y:.1f}) observed=({observed[0]:.1f},{observed[1]:.1f}) Δ=({dx:.1f},{dy:.1f})")

    # Restore cursor
    hid.move(float(start[0]), float(start[1]))
    print("Restored cursor position.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

