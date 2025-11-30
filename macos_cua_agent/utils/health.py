from __future__ import annotations

import os

from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import (
    has_accessibility_permission,
    has_screen_recording_permission,
    request_permissions,
)


def run_permission_health_checks(settings, logger=None) -> None:
    """
    Fail fast if required macOS permissions are missing.
    - Screen Recording: required for screenshots (vision pipeline).
    - Accessibility: required for HID input and Accessibility Tree reads.
    """
    logger = logger or get_logger(__name__)
    missing = []

    if not has_screen_recording_permission():
        missing.append("Screen Recording")

    if (settings.enable_hid or settings.enable_semantic) and not has_accessibility_permission():
        missing.append("Accessibility")

    if missing:
        if os.getenv("PROMPT_PERMISSIONS", "true").lower() in {"1", "true", "yes", "on"}:
            try:
                prompted = request_permissions(logger=logger)
                if prompted:
                    logger.info(
                        "Permission prompts were triggered. Approve them and restart the app to continue."
                    )
            except Exception as exc:
                logger.warning("Unable to auto-request macOS permissions: %s", exc)

        msg = (
            "Missing required macOS permissions: "
            f"{', '.join(missing)}. "
            "Grant access to your terminal/IDE under System Settings > Privacy & Security "
            "> Screen Recording and Accessibility, then restart the app."
        )
        logger.error(msg)
        raise RuntimeError(msg)
