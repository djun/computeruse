"""Helper to read token usage off an OpenAI/OpenRouter-style response."""

from __future__ import annotations

from typing import Any


def usage_tokens(response: Any) -> int:
    """Best-effort total token count for a chat completion response.

    Returns 0 when usage is missing (e.g. stubbed clients), so callers can add
    it to a running total unconditionally.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")
    try:
        return int(total or 0)
    except (TypeError, ValueError):
        return 0
