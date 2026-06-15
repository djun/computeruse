"""Helpers for building image data URIs with the actual encoded MIME type."""

from __future__ import annotations

import base64
import re


def configured_image_mime(encode_format: str | None) -> str:
    token = str(encode_format or "").strip().lower()
    if token == "png":
        return "image/png"
    if token in {"jpg", "jpeg"}:
        return "image/jpeg"
    return "image/jpeg"


def image_mime_from_base64(image_b64: str, *, fallback: str = "image/jpeg") -> str:
    """Return the actual MIME for a base64 image payload when it is sniffable."""
    raw = str(image_b64 or "").strip()
    if not raw:
        return fallback

    if raw.startswith("data:"):
        match = re.match(r"^data:([^;,]+);base64,(.*)$", raw, flags=re.DOTALL)
        if match:
            declared = match.group(1).strip()
            raw = match.group(2).strip()
            fallback = declared or fallback

    sample = re.sub(r"\s+", "", raw)[:128]
    if not sample:
        return fallback
    sample += "=" * (-len(sample) % 4)
    try:
        header = base64.b64decode(sample, validate=False)
    except Exception:
        return fallback

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def image_data_uri(image_b64: str, *, fallback: str = "image/jpeg") -> str:
    mime = image_mime_from_base64(image_b64, fallback=fallback)
    raw = str(image_b64 or "").strip()
    if raw.startswith("data:"):
        match = re.match(r"^data:[^;,]+;base64,(.*)$", raw, flags=re.DOTALL)
        if match:
            raw = match.group(1).strip()
    return f"data:{mime};base64,{raw}"
