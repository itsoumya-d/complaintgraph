"""Label and text utilities shared by training, analysis and tests."""
from __future__ import annotations

import re

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}


def star_to_label(star: int) -> int:
    """Map a 1-5 star rating to a three-class sentiment label."""
    if star is None:
        raise ValueError("star rating is required")
    star = int(star)
    if not 1 <= star <= 5:
        raise ValueError(f"star must be 1-5, got {star}")
    if star <= 2:
        return 0
    if star == 3:
        return 1
    return 2


def clean_text(text: str, max_chars: int = 512) -> str:
    """Normalise review text without destroying the complaint signal."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def englishish(text: str, threshold: float = 0.7) -> bool:
    """Cheap filter for the mostly-English public review sample.

    This is deliberately not a language detector; it only removes reviews
    that are overwhelmingly non-ASCII so the tiny model is not trained on
    mixed scripts it cannot represent well.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) >= threshold
