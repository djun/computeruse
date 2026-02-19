from __future__ import annotations

import pytest

from cua_agent.observability.dashboard import LiveDebugDashboard, _DASHBOARD_HTML
from cua_agent.utils.config import Settings


class _Logger:
    def info(self, *args, **kwargs) -> None:  # pragma: no cover - noop logger
        return None

    def warning(self, *args, **kwargs) -> None:  # pragma: no cover - noop logger
        return None


@pytest.mark.parametrize(
    ("encode_format", "expected_mime"),
    [
        ("PNG", "image/png"),
        ("jpeg", "image/jpeg"),
        ("unknown", "image/jpeg"),
    ],
)
def test_dashboard_frame_mime_tracks_encode_format(encode_format: str, expected_mime: str) -> None:
    settings = Settings(enable_debug_dashboard=False, encode_format=encode_format)
    dashboard = LiveDebugDashboard(settings, _Logger())
    assert dashboard.snapshot()["frame_mime"] == expected_mime


def test_dashboard_html_uses_dynamic_frame_mime_in_data_uri() -> None:
    assert 'const frameMime = snapshot.frame_mime || "image/jpeg";' in _DASHBOARD_HTML
    assert 'screen.src = "data:" + frameMime + ";base64," + snapshot.frame_b64;' in _DASHBOARD_HTML
