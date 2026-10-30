"""Frozen-encoder transfer learning baseline.

Why this exists: end-to-end fine-tuning of DistilBERT on CPU is slow. This
script computes mean-pooled DistilBERT embeddings once, then trains a linear
classifier on them. It is still a trained model, it is reproducible, and the
metrics are honest about what it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from .data import build_rows, split_rows

BASE_MODEL = "distilbert-base-uncased"
DEFAULT_OUT = Path("models/distilbert-frozen-logreg")


def embed_texts(texts: list[str], batch_size: int = 32, max_length: int = 96) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
    model.eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def train(limit_per_class: int = 1500, max_length: int = 96, output_dir: Path = DEFAULT_OUT, seed: int = 42) -> dict:
    rows = build_rows(limit_per_class=limit_per_class, seed=seed)
    train_rows, eval_rows = split_rows(rows, eval_ratio=0.2, seed=seed)
    x_train = embed_texts([r["text"] for r in train_rows], max_length=max_length)
    x_eval = embed_texts([r["text"] for r in eval_rows], max_length=max_length)
    y_train = np.array([r["label"] for r in train_rows])
    y_eval = np.array([r["label"] for r in eval_rows])

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_eval)
    metrics = {
        "base_model": BASE_MODEL,
        "method": "frozen mean-pooled embeddings + logistic regression",
        "accuracy": float(accuracy_score(y_eval, pred)),
        "macro_f1": float(f1_score(y_eval, pred, average="macro")),
        "per_class_f1": {
            str(i): float(v)
            for i, v in enumerate(f1_score(y_eval, pred, average=None, labels=[0, 1, 2]))
        },
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "max_length": max_length,
        "labels": {"0": "negative", "1": "neutral", "2": "positive"},
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "base_model": BASE_MODEL, "max_length": max_length}, output_dir / "classifier.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-class", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.limit_per_class, args.max_length, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
