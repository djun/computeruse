"""Dedicated grounding model (composed-agent pattern).

The strong planner/core model decides *what* to interact with (a short target
description); this cheap grounder decides *where* (pixel coordinates). Designed
for UI-TARS-1.5-7B served via OpenRouter, but works with any VLM that returns a
coordinate pair for a described element.
"""

from __future__ import annotations

import re
from typing import Optional

from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.config import Settings
from cua_agent.utils.image_mime import configured_image_mime, image_data_uri
from cua_agent.utils.logger import get_logger
from cua_agent.utils.token_usage import usage_tokens

# Matches the first "(x, y)" / "x, y" / "<point>x y</point>" style coordinate pair.
_COORD_RE = re.compile(
    r"[\(\[<]?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[\)\]>]?"
)


class GroundingModelClient:
    """Resolves a target description to logical-point coordinates via a VLM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.mime = configured_image_mime(settings.encode_format)
        self.tokens_used = 0
        self.client = self._build_client()

    def _build_client(self) -> Optional[object]:
        if not self.settings.enable_uitars_grounder:
            return None
        api_key = self.settings.uitars_api_key
        if not api_key:
            self.logger.info("UI-TARS grounder disabled: UITARS_API_KEY/OPENROUTER_API_KEY missing.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self.logger.warning("openai package unavailable for grounder: %s", exc)
            return None
        return OpenAI(base_url=self.settings.uitars_base_url, api_key=api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def locate(self, screenshot_b64: str, description: str, display: DisplayInfo) -> Optional[tuple[float, float]]:
        """Return (x, y) in logical display points for the described element, or None."""
        if not self.client or not description:
            return None

        width = int(display.logical_width)
        height = int(display.logical_height)
        prompt = (
            "You are a precise GUI grounding model. Locate the described UI element in the "
            f"screenshot (image size {width}x{height} pixels, origin top-left).\n"
            f"Element: {description}\n"
            "Respond with ONLY the click target as pixel coordinates in the form (x, y)."
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_b64, fallback=self.mime)}},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.uitars_model,
                messages=[
                    {"role": "system", "content": "Return only the click coordinates."},
                    {"role": "user", "content": content},
                ],
                max_tokens=128,
            )
            self.tokens_used += usage_tokens(response)
            raw = response.choices[0].message.content if response and response.choices else ""
            text = "".join(frag.text for frag in raw) if isinstance(raw, list) else str(raw or "")
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("UI-TARS grounding failed: %s", exc)
            return None

        return self._parse_coordinates(text, width, height)

    def _parse_coordinates(self, text: str, width: int, height: int) -> Optional[tuple[float, float]]:
        match = _COORD_RE.search(text or "")
        if not match:
            return None
        try:
            raw_x = float(match.group(1))
            raw_y = float(match.group(2))
        except (TypeError, ValueError):
            return None

        x, y = self._to_pixels(raw_x, raw_y, width, height)
        if x is None or y is None:
            return None
        # Clamp inside the frame.
        x = max(0.0, min(float(width), x))
        y = max(0.0, min(float(height), y))
        return x, y

    @staticmethod
    def _to_pixels(raw_x: float, raw_y: float, width: int, height: int) -> tuple[Optional[float], Optional[float]]:
        space = "auto"  # kept local; UI-TARS-1.5 emits absolute pixels, older emit 0-1000
        if raw_x < 0 or raw_y < 0:
            return None, None
        within_frame = raw_x <= width and raw_y <= height
        looks_normalized = raw_x <= 1000 and raw_y <= 1000 and (width > 1000 or height > 1000)
        if space == "normalized_1000" or (space == "auto" and not within_frame and looks_normalized):
            return raw_x / 1000.0 * width, raw_y / 1000.0 * height
        return raw_x, raw_y
