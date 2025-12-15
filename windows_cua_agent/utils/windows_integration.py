from __future__ import annotations

import ctypes
import platform
from typing import Optional

from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.logger import get_logger


def ensure_dpi_awareness(logger_name: str = __name__) -> None:
    """
    Attempt to set the current process as DPI-aware so Windows doesn't virtualize
    coordinates and screen metrics.

    Best-effort: failures are logged but not fatal.
    """
    logger = get_logger(logger_name)
    try:
        # Windows 10+: per-monitor v2 awareness is best
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)  # noqa: N806
            if user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
                return
        # Windows 8.1+: shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE=2)
        shcore = ctypes.windll.shcore  # type: ignore[attr-defined]
        if hasattr(shcore, "SetProcessDpiAwareness"):
            PROCESS_PER_MONITOR_DPI_AWARE = 2  # noqa: N806
            shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
            return
        # Vista+: system DPI aware fallback
        if hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
            return
    except Exception as exc:
        logger.debug("DPI awareness setup failed (non-fatal): %s", exc)


def get_display_info() -> DisplayInfo:
    """
    Return primary display info with best-effort DPI scale factor.

    `logical_*` represent the coordinate space used by the agent/model.
    `physical_*` represent the true framebuffer size in pixels.
    """
    logger = get_logger(__name__)

    # Ensure DPI awareness so GetSystemMetrics returns physical pixels.
    ensure_dpi_awareness(logger_name=__name__)

    physical = _get_primary_monitor_physical_size()
    scale = _get_primary_monitor_scale_factor()

    if physical and scale:
        physical_w, physical_h = physical
        logical_w = int(round(physical_w / scale))
        logical_h = int(round(physical_h / scale))
        return DisplayInfo(
            logical_width=logical_w,
            logical_height=logical_h,
            physical_width=physical_w,
            physical_height=physical_h,
            scale_factor=scale,
        )

    # Fallback: pyautogui (already in requirements) returns logical size in most cases.
    try:
        import pyautogui  # type: ignore

        w, h = pyautogui.size()
        return DisplayInfo(
            logical_width=int(w),
            logical_height=int(h),
            physical_width=int(w),
            physical_height=int(h),
            scale_factor=1.0,
        )
    except Exception:
        logger.warning("Falling back to default display info (1280x720, scale=1.0).")
        return DisplayInfo(
            logical_width=1280,
            logical_height=720,
            physical_width=1280,
            physical_height=720,
            scale_factor=1.0,
        )


def _get_primary_monitor_physical_size() -> Optional[tuple[int, int]]:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        SM_CXSCREEN = 0
        SM_CYSCREEN = 1
        return int(user32.GetSystemMetrics(SM_CXSCREEN)), int(user32.GetSystemMetrics(SM_CYSCREEN))
    except Exception:
        return None


def _get_primary_monitor_scale_factor() -> Optional[float]:
    """
    Best-effort primary monitor scale factor.
    """
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if hasattr(user32, "GetDpiForSystem"):
            dpi = int(user32.GetDpiForSystem())
            if dpi > 0:
                return float(dpi) / 96.0
    except Exception:
        pass

    try:
        # Per-monitor scaling via shcore.GetScaleFactorForMonitor
        shcore = ctypes.windll.shcore  # type: ignore[attr-defined]
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]

        class POINT(ctypes.Structure):  # noqa: D401
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        MONITOR_DEFAULTTOPRIMARY = 1
        monitor = user32.MonitorFromPoint(POINT(0, 0), MONITOR_DEFAULTTOPRIMARY)
        if not monitor:
            return None

        scale_pct = ctypes.c_int()
        if shcore.GetScaleFactorForMonitor(monitor, ctypes.byref(scale_pct)) == 0 and scale_pct.value:
            return float(scale_pct.value) / 100.0
    except Exception:
        pass

    return None


def get_system_info() -> str:
    try:
        ver = platform.version()
        release = platform.release()
        arch = platform.machine()
        return f"Windows {release} ({ver}) [{arch}]"
    except Exception:
        return "Windows (Unknown System)"


def get_foreground_window_title() -> str:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value or ""
    except Exception:
        return ""


def get_foreground_process_image_name() -> str:
    """
    Return the image name of the process owning the foreground window (e.g., "explorer.exe").
    Best-effort; returns empty string on failure.
    """
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""

        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return ""
        try:
            # QueryFullProcessImageNameW returns full path; we take basename.
            buf_len = ctypes.c_ulong(260)
            buf = ctypes.create_unicode_buffer(buf_len.value)
            if hasattr(kernel32, "QueryFullProcessImageNameW"):
                if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(buf_len)):
                    path = buf.value
                    return path.split("\\")[-1].lower()
            # Fallback to GetModuleBaseNameW (requires psapi)
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            if psapi.GetModuleBaseNameW(hproc, None, buf, buf_len.value):
                return (buf.value or "").lower()
        finally:
            kernel32.CloseHandle(hproc)
    except Exception:
        return ""

    return ""
