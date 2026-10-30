# Model Card — ComplaintGraph review sentiment (bert-tiny)

## Model details

- **Base model:** [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny)
- **Task:** 3-class review sentiment (negative / neutral / positive)
- **Labels:** 1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive
- **Output:** class probabilities; the report uses the negative class as the complaint signal
- **Training code:** `src/complaintgraph/train.py`
- **Artifact:** `models/bert-tiny-app-review-sentiment/` (not committed; reproducible)

## Training data

Public `app_reviews` dataset (Google Play reviews with package name, review text, date and star rating).
Training uses a balanced sample per class after cleaning, URL/HTML removal and a cheap mostly-English filter.
No private data, no company-internal data.

## Metrics

The committed artifact is the frozen-encoder transfer-learning baseline in `models/distilbert-frozen-logreg/`
(not committed) with metrics copied to `metrics/train_metrics.json`. An end-to-end `bert-tiny` fine-tune is also
implemented in `train.py`; the one-epoch CPU run reached 0.532 accuracy and is kept as a lightweight baseline, not
the headline model.

| Metric | Value |
|---|---|
| Accuracy | **0.690** |
| Macro F1 | **0.689** |
| Train / eval size | 2,880 / 720 |
| Method | Frozen mean-pooled DistilBERT embeddings + logistic regression |
| Per-class F1 | negative 0.696 · neutral 0.577 · positive 0.793 |

## Intended use

- Directional triage of public review complaints.
- Theme discovery before a product research pass.
- A reproducible baseline that can be replaced with a larger model.

## Out of scope

- Production moderation or automated customer decisions.
- Claims about a company's internal metrics, revenue or churn.
- Non-English reviews at scale (the tiny model and sample are English-heavy).
- Any use that treats cluster labels as causal evidence.

## Limitations and bias

- Star ratings are a noisy proxy; sarcasm, mixed reviews and rating inflation all leak in.
- App-review populations skew toward highly motivated users.
- `bert-tiny` is a small model; it will miss nuance and long-range context.
- The mostly-English filter excludes many markets, which matters for global products.
- Clusters are descriptive. Suggested experiments must be validated with product analytics and user research.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m complaintgraph.frozen --limit-per-class 1200 --max-length 96
# optional end-to-end fine-tune:
python -m complaintgraph.train --limit-per-class 2000 --epochs 1
```
