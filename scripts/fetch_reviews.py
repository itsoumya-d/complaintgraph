"""Fetch public Google Play reviews for a package.

Uses google-play-scraper when installed; otherwise falls back to the public
`app_reviews` dataset (which may not contain every package).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def fetch_scraper(package: str, count: int):
    from google_play_scraper import Sort, reviews

    rows, _ = reviews(package, count=count, sort=Sort.NEWEST, lang="en", country="us")
    return [{"review": r.get("content", ""), "star": r.get("score", 0), "date": str(r.get("at", ""))} for r in rows]


def fetch_dataset(package: str, count: int):
    from datasets import load_dataset

    ds = load_dataset("app_reviews", split="train")
    rows = []
    for row in ds:
        if row.get("package_name") != package:
            continue
        rows.append({"review": row.get("review", ""), "star": row.get("star", 0), "date": row.get("date", "")})
        if len(rows) >= count:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, help="e.g. com.sumup.app")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        rows = fetch_scraper(args.package, args.count)
    except ImportError:
        rows = fetch_dataset(args.package, args.count)
    if not rows:
        raise SystemExit(
            f"no reviews found for {args.package}. Install google-play-scraper (`pip install google-play-scraper`) "
            "or choose a package present in the public app_reviews dataset."
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["review", "star", "date"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} reviews)")


if __name__ == "__main__":
    main()
