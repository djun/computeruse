from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from PIL import Image, ImageFilter

# Brazilian CPF patterns (with and without punctuation)
CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")

# Common secret labels in PT/EN and security-related tokens.
SECRET_KEYWORD_PATTERN = re.compile(
    r"(?i)\b(password|senha|passcode|secret|token|api[_ -]?key|otp|pin|cvv|cvc)\b"
)

SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"
)


def redact_sensitive_regions(
    image: Image.Image,
    *,
    min_confidence: float = 35.0,
    blur_padding_px: int = 4,
) -> tuple[Image.Image, int]:
    """
    Detect likely sensitive text boxes via OCR and blur matching regions.

    Returns (possibly-redacted image, number_of_redacted_regions).
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return image, 0

    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return image, 0

    words = _extract_words(ocr_data, min_confidence=min_confidence)
    if not words:
        return image, 0

    sensitive_indices = _find_sensitive_word_indices(words)
    if not sensitive_indices:
        return image, 0

    regions = _indices_to_regions(
        words,
        sensitive_indices,
        image_width=image.width,
        image_height=image.height,
        padding=max(0, int(blur_padding_px)),
    )
    if not regions:
        return image, 0

    merged_regions = _merge_regions(regions, gap=max(2, int(blur_padding_px)))
    redacted = _blur_regions(image, merged_regions)
    return redacted, len(merged_regions)


def _extract_words(ocr_data: dict, *, min_confidence: float) -> list[dict[str, Any]]:
    texts = ocr_data.get("text") or []
    n = len(texts)
    confs = ocr_data.get("conf") or [0] * n
    lefts = ocr_data.get("left") or [0] * n
    tops = ocr_data.get("top") or [0] * n
    widths = ocr_data.get("width") or [0] * n
    heights = ocr_data.get("height") or [0] * n
    blocks = ocr_data.get("block_num") or [0] * n
    pars = ocr_data.get("par_num") or [0] * n
    lines = ocr_data.get("line_num") or [0] * n

    words: list[dict[str, Any]] = []
    for i in range(n):
        text = str(texts[i] or "").strip()
        if not text:
            continue
        try:
            conf = float(confs[i] or 0)
        except Exception:
            conf = 0.0
        if conf < min_confidence:
            continue
        try:
            x = int(lefts[i] or 0)
            y = int(tops[i] or 0)
            w = int(widths[i] or 0)
            h = int(heights[i] or 0)
        except Exception:
            continue
        if w <= 1 or h <= 1:
            continue
        words.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "block": int(blocks[i] or 0),
                "par": int(pars[i] or 0),
                "line": int(lines[i] or 0),
            }
        )
    return words


def _find_sensitive_word_indices(words: list[dict[str, Any]]) -> set[int]:
    sensitive: set[int] = set()
    lines: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for idx, word in enumerate(words):
        line_key = (word["block"], word["par"], word["line"])
        lines[line_key].append(idx)

    for indices in lines.values():
        line_text = " ".join(words[i]["text"] for i in indices)
        line_lower = line_text.lower()
        line_is_sensitive = (
            _contains_sensitive_pair(line_lower)
            or _contains_probable_cpf(line_text)
            or _contains_probable_credit_card(line_text)
            or _contains_secret_value(line_text)
        )
        if line_is_sensitive:
            sensitive.update(indices)
            continue

        for pos, word_idx in enumerate(indices):
            token = words[word_idx]["text"]
            token_lower = token.lower()
            if _is_sensitive_token(token):
                sensitive.add(word_idx)
                continue
            if SECRET_KEYWORD_PATTERN.search(token_lower):
                sensitive.add(word_idx)
                for near in (pos + 1, pos + 2):
                    if near < len(indices):
                        sensitive.add(indices[near])

    return sensitive


def _contains_sensitive_pair(line_lower: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(password|senha|passcode|secret|token|api[_ -]?key|otp|pin)\b\s*[:=]?",
            line_lower,
        )
    )


def _contains_probable_cpf(text: str) -> bool:
    return bool(CPF_PATTERN.search(text))


def _contains_secret_value(text: str) -> bool:
    return bool(SECRET_VALUE_PATTERN.search(text))


def _contains_probable_credit_card(text: str) -> bool:
    # Digits separated by spaces/hyphens; validate with Luhn to reduce false positives.
    for candidate in re.findall(r"(?:\d[ -]?){13,23}", text):
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return True
    return False


def _is_sensitive_token(token: str) -> bool:
    if not token:
        return False
    if _contains_probable_cpf(token):
        return True
    if _contains_secret_value(token):
        return True
    if _contains_probable_credit_card(token):
        return True
    return _is_high_entropy_secret(token)


def _is_high_entropy_secret(token: str) -> bool:
    cleaned = token.strip()
    if len(cleaned) < 20:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_+=/\-]+", cleaned):
        return False
    if not re.search(r"[A-Za-z]", cleaned) or not re.search(r"\d", cleaned):
        return False
    return _shannon_entropy(cleaned) >= 3.8


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    freq = {ch: value.count(ch) for ch in set(value)}
    total = float(len(value))
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


def _luhn_valid(number: str) -> bool:
    if not number.isdigit():
        return False
    total = 0
    parity = len(number) % 2
    for idx, ch in enumerate(number):
        digit = int(ch)
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (total % 10) == 0


def _indices_to_regions(
    words: list[dict[str, Any]],
    indices: set[int],
    *,
    image_width: int,
    image_height: int,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    regions: list[tuple[int, int, int, int]] = []
    for idx in indices:
        word = words[idx]
        x0 = max(0, int(word["x"]) - padding)
        y0 = max(0, int(word["y"]) - padding)
        x1 = min(image_width, int(word["x"]) + int(word["w"]) + padding)
        y1 = min(image_height, int(word["y"]) + int(word["h"]) + padding)
        if x1 > x0 and y1 > y0:
            regions.append((x0, y0, x1, y1))
    return regions


def _merge_regions(
    regions: list[tuple[int, int, int, int]],
    *,
    gap: int,
) -> list[tuple[int, int, int, int]]:
    if not regions:
        return []

    pending = sorted(regions, key=lambda r: (r[1], r[0]))
    merged: list[tuple[int, int, int, int]] = []

    for x0, y0, x1, y1 in pending:
        merged_any = False
        for idx, (mx0, my0, mx1, my1) in enumerate(merged):
            separated = x1 < (mx0 - gap) or mx1 < (x0 - gap) or y1 < (my0 - gap) or my1 < (y0 - gap)
            if separated:
                continue
            merged[idx] = (min(mx0, x0), min(my0, y0), max(mx1, x1), max(my1, y1))
            merged_any = True
            break
        if not merged_any:
            merged.append((x0, y0, x1, y1))
    return merged


def _blur_regions(image: Image.Image, regions: list[tuple[int, int, int, int]]) -> Image.Image:
    if not regions:
        return image
    redacted = image.copy()
    for x0, y0, x1, y1 in regions:
        crop = redacted.crop((x0, y0, x1, y1))
        radius = max(2.0, min((x1 - x0), (y1 - y0)) / 8.0)
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=radius))
        redacted.paste(blurred, (x0, y0))
    return redacted
