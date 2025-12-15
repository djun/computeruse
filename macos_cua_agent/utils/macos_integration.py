from __future__ import annotations

from typing import Optional

from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.logger import get_logger

logger = get_logger(__name__)


def get_display_info() -> DisplayInfo:
    """Return display info with graceful fallbacks when pyobjc is unavailable."""
    info = _from_pyobjc()
    if info:
        return info

    info = _from_pyautogui()
    if info:
        return info

    logger.warning("Falling back to default display info (1280x720, scale=1.0).")
    return DisplayInfo(
        logical_width=1280,
        logical_height=720,
        physical_width=1280,
        physical_height=720,
        scale_factor=1.0,
    )


def _from_pyobjc() -> Optional[DisplayInfo]:
    try:
        from AppKit import NSScreen  # type: ignore
    except Exception:
        return None

    screen = NSScreen.mainScreen()
    if not screen:
        return None

    frame = screen.frame()
    logical_width = int(frame.size.width)
    logical_height = int(frame.size.height)
    scale = float(screen.backingScaleFactor()) if hasattr(screen, "backingScaleFactor") else 1.0
    return DisplayInfo(
        logical_width=logical_width,
        logical_height=logical_height,
        physical_width=int(logical_width * scale),
        physical_height=int(logical_height * scale),
        scale_factor=scale,
    )


def _from_pyautogui() -> Optional[DisplayInfo]:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return None

    width, height = pyautogui.size()
    return DisplayInfo(
        logical_width=int(width),
        logical_height=int(height),
        physical_width=int(width),
        physical_height=int(height),
        scale_factor=1.0,
    )


def has_screen_recording_permission() -> bool:
    try:
        import Quartz  # type: ignore

        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def has_accessibility_permission() -> bool:
    ax_is_trusted, _, _ = _get_accessibility_api()
    if not ax_is_trusted:
        logger.warning("Accessibility API unavailable; returning not trusted.")
        return False

    try:
        return bool(ax_is_trusted())
    except Exception:
        return False


def request_permissions(logger=None) -> bool:
    """
    Ask macOS to show permission prompts for Screen Recording and Accessibility.
    Returns True if a prompt was requested.
    """
    logger = logger or get_logger(__name__)
    try:
        import Quartz  # type: ignore
    except Exception as exc:
        logger.warning("Quartz unavailable; cannot request permissions: %s", exc)
        return False

    prompted = False

    ax_is_trusted, ax_prompt, ax_prompt_opt = _get_accessibility_api()

    try:
        if not Quartz.CGPreflightScreenCaptureAccess():
            Quartz.CGRequestScreenCaptureAccess()
            prompted = True
    except Exception as exc:
        logger.warning("Failed to request Screen Recording permission: %s", exc)

    try:
        if ax_is_trusted and ax_prompt and ax_prompt_opt:
            if not ax_is_trusted():
                ax_prompt({ax_prompt_opt: True})
                prompted = True
        elif not ax_is_trusted:
            logger.warning("Accessibility API unavailable; cannot request permission.")
    except Exception as exc:
        logger.warning("Failed to request Accessibility permission: %s", exc)

    if prompted:
        logger.info("Requested macOS permission prompts; approve them, then restart this app.")

    return prompted


def _get_accessibility_api():
    """
    Returns (AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt)
    trying ApplicationServices first, then Quartz.
    """
    try:
        import ApplicationServices  # type: ignore

        ax_is_trusted = getattr(ApplicationServices, "AXIsProcessTrusted", None)
        ax_prompt = getattr(ApplicationServices, "AXIsProcessTrustedWithOptions", None)
        ax_prompt_opt = getattr(ApplicationServices, "kAXTrustedCheckOptionPrompt", None)
        if ax_is_trusted:
            return ax_is_trusted, ax_prompt, ax_prompt_opt
    except Exception:
        pass

    try:
        import Quartz  # type: ignore

        ax_is_trusted = getattr(Quartz, "AXIsProcessTrusted", None)
        ax_prompt = getattr(Quartz, "AXIsProcessTrustedWithOptions", None)
        ax_prompt_opt = getattr(Quartz, "kAXTrustedCheckOptionPrompt", None)
        if ax_is_trusted:
            return ax_is_trusted, ax_prompt, ax_prompt_opt
    except Exception:
        pass

    return None, None, None


def get_system_info() -> str:
    """Returns a string describing the macOS version and hardware model."""
    import platform
    import subprocess

    try:
        ver, _, arch = platform.mac_ver()
        model = "Unknown Mac"
        try:
            # sysctl -n hw.model returns something like "Mac14,2" or "MacBookPro18,3"
            res = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True)
            model = res.strip()
        except Exception:
            pass

        return f"macOS {ver} ({arch}) on {model}"
    except Exception:
        return "macOS (Unknown System)"
