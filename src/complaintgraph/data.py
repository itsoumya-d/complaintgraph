"""Dataset loading and deterministic sampling for training and reports."""
from __future__ import annotations

import random
from typing import Iterable

from datasets import load_dataset

from .labels import clean_text, englishish, star_to_label


def load_app_reviews():
    """Load the public `app_reviews` dataset (Google Play reviews with stars)."""
    return load_dataset("app_reviews", split="train")


def build_rows(limit_per_class: int = 2000, seed: int = 42) -> list[dict]:
    """Collect a balanced, English-ish sample from the public review dataset.

    Each row is `{text, label, star, package}`. Labels are derived from the
    star rating: 1-2 negative, 3 neutral, 4-5 positive.
    """
    ds = load_app_reviews()
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for row in ds:
        text = clean_text(row.get("review", ""))
        if len(text) < 20 or not englishish(text):
            continue
        try:
            label = star_to_label(row.get("star"))
        except (TypeError, ValueError):
            continue
        if len(buckets[label]) >= limit_per_class:
            if all(len(v) >= limit_per_class for v in buckets.values()):
                break
            continue
        buckets[label].append({
            "text": text,
            "label": label,
            "star": int(row["star"]),
            "package": row.get("package_name", ""),
        })
    rows: list[dict] = []
    for label, items in buckets.items():
        rng.shuffle(items)
        rows.extend(items[:limit_per_class])
    rng.shuffle(rows)
    return rows


def split_rows(rows: Iterable[dict], eval_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified split so every label appears in both train and eval."""
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    train: list[dict] = []
    evalset: list[dict] = []
    for items in by_label.values():
        rng.shuffle(items)
        cut = max(1, int(len(items) * eval_ratio)) if len(items) > 1 else 0
        evalset.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(evalset)
    return train, evalset
