from __future__ import annotations

import argparse
import sys
import time

from cua_agent.utils.config import Settings
from windows_cua_agent.drivers.hid_driver import HIDDriver


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hid_smoke_test",
        description="Basic Windows SendInput smoke test (requires explicit confirmation).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually send input (moves mouse/keyboard). Requires ENABLE_HID=true.",
    )
    parser.add_argument(
        "--app",
        default="Notepad",
        help="App name to launch via the Start menu (default: Notepad).",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=2.0,
        help="Countdown before input begins (seconds).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    settings = Settings()
    armed = bool(args.confirm) and bool(settings.enable_hid)

    if not settings.enable_hid:
        print("ENABLE_HID is false; driver will run in dry-run mode.")
        print("Set ENABLE_HID=true (in your environment/.env) and re-run with --confirm to execute.")

    if args.confirm and not settings.enable_hid:
        print("Refusing to run: --confirm requires ENABLE_HID=true.")
        return 2

    # Force dry-run unless explicitly armed.
    settings.enable_hid = armed
    hid = HIDDriver(settings)

    print("Planned actions:")
    print("- Press Win")
    print(f"- Type {args.app!r}")
    print("- Press Enter")
    if not armed:
        print("Mode: dry-run (no input will be sent).")
        return 0

    if args.delay_s > 0:
        print(f"Starting in {args.delay_s:.1f}s...")
        time.sleep(float(args.delay_s))

    hid.press_keys(["win"])
    time.sleep(0.4)
    hid.type_text(args.app)
    time.sleep(0.2)
    hid.press_keys(["enter"])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

