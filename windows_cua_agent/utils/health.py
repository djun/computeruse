from __future__ import annotations

import os
import subprocess
import sys
import socket
from dataclasses import dataclass
from typing import Any

from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from windows_cua_agent.utils.windows_integration import ensure_dpi_awareness, get_display_info


@dataclass(frozen=True)
class WindowsHealthInfo:
    integrity: str
    elevated: bool


def _attempt_relaunch_as_admin(logger) -> None:
    """
    Best-effort UAC prompt by relaunching the current Python process with the `runas` verb.

    Raises SystemExit(0) after successfully spawning the elevated process.
    """
    if os.environ.get("CUA_ELEVATION_ATTEMPTED") == "1":
        raise RuntimeError("Elevation already requested but process is still not elevated.")

    try:
        import ctypes
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to request elevation: {exc}") from exc

    os.environ["CUA_ELEVATION_ATTEMPTED"] = "1"

    args = _build_elevation_relaunch_args()
    params = subprocess.list2cmdline(args)
    exe = sys.executable
    cwd = os.getcwd()

    logger.warning("Requesting Administrator privileges (UAC prompt) to enable Windows HID automation...")
    rc = int(ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, params, cwd, 1))  # type: ignore[attr-defined]
    if rc <= 32:
        raise RuntimeError(
            "Elevation was cancelled or failed. Re-run from an Administrator shell, or set ENABLE_HID=false."
        )
    raise SystemExit(0)


def _build_elevation_relaunch_args() -> list[str]:
    main_mod = sys.modules.get("__main__")
    spec = getattr(main_mod, "__spec__", None)
    if spec and getattr(spec, "name", None):
        return ["-m", str(spec.name), *sys.argv[1:]]

    script = sys.argv[0] if sys.argv else ""
    if script.endswith(".py"):
        abs_script = os.path.abspath(script)
        try:
            rel = os.path.relpath(abs_script, os.getcwd())
        except Exception:
            rel = ""
        if rel and not rel.startswith("..") and rel.lower().endswith(".py"):
            module_guess = rel[:-3].replace(os.sep, ".")
            return ["-m", module_guess, *sys.argv[1:]]

    return list(sys.argv)


def run_permission_health_checks(settings: Settings, logger: Any | None = None) -> None:  # noqa: ARG001
    """
    Best-effort health checks for Windows.

    Windows doesn't have a macOS-style permission prompt flow for screen capture and input,
    but DPI virtualization and privilege boundaries can still break automation.
    """
    logger = logger or get_logger(__name__)

    try:
        ensure_dpi_awareness(logger_name=__name__)
    except Exception as exc:
        logger.debug("DPI awareness check failed (non-fatal): %s", exc)

    _check_capture_alignment(logger)

    info = _get_windows_health_info(logger)
    if info:
        logger.info("Windows integrity=%s elevated=%s", info.integrity, info.elevated)

        # Low/untrusted integrity can break cross-process UIA/HID behavior.
        if info.integrity in {"untrusted", "low"} and (settings.enable_hid or settings.enable_semantic):
            raise RuntimeError(
                "Process is running at low/untrusted integrity; Windows UI automation may be blocked. "
                "Run from a normal desktop session (medium integrity) and try again."
            )

        # If the agent is not elevated, it cannot send input to elevated windows (UIPI).
        if settings.enable_hid and not info.elevated:
            if getattr(settings, "windows_auto_elevate", True):
                _attempt_relaunch_as_admin(logger)
            logger.warning(
                "ENABLE_HID is true but process is not elevated; interactions with admin/UAC windows may fail (UIPI). "
                "Re-run as Administrator or set WINDOWS_AUTO_ELEVATE=true to prompt."
            )

    # Browser automation uses Chrome DevTools Protocol (CDP) on localhost.
    # This is optional; just provide a helpful hint.
    if not _is_tcp_port_open("127.0.0.1", 9222, timeout_s=0.2):
        logger.debug(
            "CDP not detected on 127.0.0.1:9222. To enable browser automation, launch Chrome/Edge with "
            "--remote-debugging-port=9222."
        )


def _check_capture_alignment(logger) -> None:
    """
    Best-effort validation that screen metrics and capture backends agree.

    A mismatch is a strong signal that DPI virtualization is active and can cause
    coordinate misalignment between VisionPipeline and HID injection.
    """
    try:
        display = get_display_info()
    except Exception as exc:
        logger.debug("Display probe failed (non-fatal): %s", exc)
        return

    try:
        import mss  # type: ignore

        with mss.mss(with_cursor=False) as sct:
            mon = sct.monitors[1]  # primary
            mss_w, mss_h = int(mon.get("width", 0)), int(mon.get("height", 0))
    except Exception as exc:
        logger.debug("mss monitor probe failed (non-fatal): %s", exc)
        return

    phys_w, phys_h = int(display.physical_width), int(display.physical_height)
    if not phys_w or not phys_h or not mss_w or not mss_h:
        return

    if (mss_w, mss_h) != (phys_w, phys_h):
        logger.warning(
            "Display metric mismatch (possible DPI virtualization): GetSystemMetrics=%sx%s vs mss=%sx%s",
            phys_w,
            phys_h,
            mss_w,
            mss_h,
        )


def _is_tcp_port_open(host: str, port: int, *, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except Exception:
        return False


def _get_windows_health_info(logger) -> WindowsHealthInfo | None:
    try:
        integrity = _get_process_integrity_level() or "unknown"
        elevated = bool(_is_process_elevated())
        return WindowsHealthInfo(integrity=integrity, elevated=elevated)
    except Exception as exc:
        logger.debug("Health info probe failed (non-fatal): %s", exc)
        return None


def _configure_win32_token_ffi(advapi32, kernel32) -> None:
    import ctypes
    from ctypes import wintypes

    advapi32.OpenProcessToken.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = wintypes.BOOL

    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = wintypes.BOOL

    advapi32.ConvertSidToStringSidW.argtypes = (
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    )
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
    kernel32.LocalFree.restype = wintypes.HLOCAL


def _is_process_elevated() -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        _configure_win32_token_ffi(advapi32, kernel32)

        TOKEN_QUERY = 0x0008
        TokenElevation = 20

        class TOKEN_ELEVATION(ctypes.Structure):
            _fields_ = [("TokenIsElevated", wintypes.DWORD)]

        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
            return False
        try:
            elev = TOKEN_ELEVATION()
            size = wintypes.DWORD(ctypes.sizeof(elev))
            if not advapi32.GetTokenInformation(
                token,
                TokenElevation,
                ctypes.byref(elev),
                size,
                ctypes.byref(size),
            ):
                return False
            return bool(elev.TokenIsElevated)
        finally:
            kernel32.CloseHandle(token)
    except Exception:
        return False


def _get_process_integrity_level() -> str | None:
    """
    Return the process integrity label: untrusted/low/medium/high/system/protected/unknown.
    """
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        TOKEN_QUERY = 0x0008
        TokenIntegrityLevel = 25

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

        class TOKEN_MANDATORY_LABEL(ctypes.Structure):
            _fields_ = [("Label", SID_AND_ATTRIBUTES)]

        _configure_win32_token_ffi(advapi32, kernel32)

        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
            return None
        try:
            needed = wintypes.DWORD(0)
            advapi32.GetTokenInformation(token, TokenIntegrityLevel, None, 0, ctypes.byref(needed))
            if needed.value <= 0:
                return None

            buf = ctypes.create_string_buffer(needed.value)
            if not advapi32.GetTokenInformation(
                token, TokenIntegrityLevel, buf, needed, ctypes.byref(needed)
            ):
                return None

            tml = TOKEN_MANDATORY_LABEL.from_buffer(buf)
            sid_ptr = tml.Label.Sid
            if not sid_ptr:
                return None

            sid_str_ptr = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(sid_ptr, ctypes.byref(sid_str_ptr)):
                return None
            try:
                sid_str = sid_str_ptr.value or ""
            finally:
                kernel32.LocalFree(sid_str_ptr)

            # Integrity SIDs are like: S-1-16-8192 (medium). We use the RID.
            rid = int(sid_str.rsplit("-", 1)[-1]) if sid_str else 0
            if rid >= 0x5000:
                return "protected"
            if rid >= 0x4000:
                return "system"
            if rid >= 0x3000:
                return "high"
            if rid >= 0x2000:
                return "medium"
            if rid >= 0x1000:
                return "low"
            return "untrusted"
        finally:
            kernel32.CloseHandle(token)
    except Exception:
        return None
