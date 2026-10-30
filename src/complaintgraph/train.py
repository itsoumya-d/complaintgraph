"""Fine-tune a tiny transformer for review sentiment, then persist metrics.

This is intentionally a small, reproducible baseline, not a claim of SOTA.
The point is a real trained artifact that can be inspected and improved.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import build_rows, split_rows

BASE_MODEL = "prajjwal1/bert-tiny"
DEFAULT_OUT = Path("models/bert-tiny-app-review-sentiment")


class ReviewDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=96):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {
            str(i): float(v)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2]))
        },
    }


def evaluate(model, loader) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            y_pred.extend(logits.argmax(dim=-1).tolist())
            y_true.extend(batch["labels"].tolist())
    return compute_metrics(y_true, y_pred)


def train(
    smoke: bool = False,
    base_model: str = BASE_MODEL,
    limit_per_class: int = 2000,
    epochs: int = 1,
    max_length: int = 96,
    batch_size: int = 32,
    lr: float = 5e-5,
    output_dir: Path = DEFAULT_OUT,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if smoke:
        limit_per_class = min(limit_per_class, 128)
        max_length = min(max_length, 64)
        batch_size = 16

    rows = build_rows(limit_per_class=limit_per_class, seed=seed)
    train_rows, eval_rows = split_rows(rows, eval_ratio=0.2, seed=seed)
    if not train_rows or not eval_rows:
        raise RuntimeError("not enough data after filtering; lower the threshold or check the dataset")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    train_loader = DataLoader(ReviewDataset(train_rows, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(ReviewDataset(eval_rows, tokenizer, max_length), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            out.loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step} loss {out.loss.item():.4f}", flush=True)

    metrics = evaluate(model, eval_loader)
    metrics.update({
        "base_model": base_model,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "epochs": epochs,
        "max_length": max_length,
        "labels": {"0": "negative", "1": "neutral", "2": "positive"},
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run for CI")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--limit-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        smoke=args.smoke,
        base_model=args.base_model,
        limit_per_class=args.limit_per_class,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
