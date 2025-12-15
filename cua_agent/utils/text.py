"""Lightweight text helpers."""

from __future__ import annotations

import re
from typing import Set


def tokenize_lower(text: str) -> Set[str]:
    """Lightweight tokenizer for similarity scoring."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(tokens)
