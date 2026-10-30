"""Generate a gap report from a CSV of public reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .labels import clean_text, star_to_label
from .report import build_report
from .themes import cluster_complaints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with review/text and star/rating columns")
    parser.add_argument("--app", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--metrics", default="models/bert-tiny-app-review-sentiment/metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    review_col = next((c for c in ("review", "text", "content") if c in df.columns), None)
    star_col = next((c for c in ("star", "rating", "score") if c in df.columns), None)
    if not review_col or not star_col:
        raise SystemExit(f"need review/text and star/rating columns; got {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        text = clean_text(str(row[review_col]))
        if len(text) < 10:
            continue
        try:
            star = int(row[star_col])
            label = star_to_label(star)
        except (TypeError, ValueError):
            continue
        rows.append({"text": text, "star": star, "label": label})

    negative = [r["text"] for r in rows if r["label"] == 0]
    themes = cluster_complaints(negative or [r["text"] for r in rows], n_clusters=args.clusters)

    metrics = None
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    report = build_report(args.app, rows, themes, metrics=metrics)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out} ({len(rows)} reviews, {len(themes)} themes)")


if __name__ == "__main__":
    main()
