"""Markdown gap-report generation from public review clusters."""
from __future__ import annotations

from datetime import datetime, timezone

from .labels import LABELS
from .themes import rating_distribution

SURFACE_EXPERIMENTS = {
    "Mobile conversation workflow": "Profile the conversation list and thread load, add skeleton/retry states, offline drafts and push-to-thread; measure time-to-first-conversation.",
    "Onboarding / authentication": "Instrument OAuth/OTP failure reasons, add a fallback sign-in path, and measure login success rate before/after.",
    "Reliability / performance": "Gate the change behind a staged rollout, track crash-free sessions, and keep a one-tap rollback path.",
    "Billing / payments": "Trace the payment state machine end to end, add idempotency keys and a visible retry/receipt state, then measure failed-payment contacts.",
    "Release / upgrade path": "Add upgrade telemetry, a staged rollout and a rollback path; measure launch failures by app version and device.",
    "Support / escalation": "Add in-app status and response-time expectations; measure repeat contacts and time-to-resolution.",
    "Search / discovery": "Instrument zero-result and low-click queries; add ranking/filter fixes and measure task completion.",
    "Notifications": "Audit notification fan-out and dedupe, add per-conversation mute/clear, and measure opt-outs and notification-driven opens.",
    "Core workflow": "Profile the critical path, add loading/retry states, and measure time-to-first-value and task completion.",
}

SURFACES = [
    (("login", "password", "otp", "signup", "sign up", "account", "google"), "Onboarding / authentication"),
    (("conversation", "chat", "ticket", "inbox"), "Mobile conversation workflow"),
    (("crash", "freeze", "slow", "lag", "loading", "performance", "battery"), "Reliability / performance"),
    (("payment", "refund", "charge", "billing", "invoice", "subscription"), "Billing / payments"),
    (("update", "version", "release", "install"), "Release / upgrade path"),
    (("support", "chat", "help", "response"), "Support / escalation"),
    (("search", "filter", "find", "sort"), "Search / discovery"),
    (("notification", "alert", "reminder"), "Notifications"),
]


def theme_name(top_terms: list[str]) -> str:
    blob = " ".join(top_terms).lower()
    if "login" in blob or "google" in blob or "sign" in blob or "password" in blob or "account" in blob:
        return "Sign-in & account access"
    if "conversation" in blob or "chat" in blob or "ticket" in blob or "inbox" in blob:
        return "Mobile conversation workflow"
    if "notification" in blob or "alert" in blob:
        return "Notifications & alerts"
    if "update" in blob or "version" in blob or "release" in blob:
        return "Update & release stability"
    if "load" in blob or "loading" in blob or "slow" in blob:
        return "Mobile loading & performance"
    if "open" in blob or "crash" in blob or "launch" in blob:
        return "Launch & crash reliability"
    if "search" in blob or "filter" in blob:
        return "Search & discovery"
    if "payment" in blob or "refund" in blob or "billing" in blob:
        return "Billing & payments"
    return "Core workflow"


def infer_surface(top_terms: list[str]) -> str:
    blob = " ".join(top_terms).lower()
    for keywords, surface in SURFACES:
        if any(k in blob for k in keywords):
            return surface
    return "Core workflow"


def build_report(
    app_name: str,
    rows: list[dict],
    themes: list[dict],
    metrics: dict | None = None,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    labels = [int(r.get("label", 0)) for r in rows]
    label_counts = {name: labels.count(i) for i, name in LABELS.items()}
    stars = rating_distribution([int(r["star"]) for r in rows if r.get("star")])

    lines = [
        f"# ComplaintGraph — {app_name}",
        "",
        f"*Generated {generated_at} from {len(rows)} public reviews. This is an independent analysis, not an official company document.*",
        "",
        "## Snapshot",
        "",
        f"- Reviews analysed: **{len(rows)}**",
        f"- Star-derived sentiment: **negative {label_counts['negative']}**, neutral {label_counts['neutral']}, positive {label_counts['positive']}",
        f"- Star distribution: " + ", ".join(f"{k}★ {v}" for k, v in stars.items()),
        "",
    ]
    if metrics:
        lines += [
            "## Model",
            "",
            f"- Base: `{metrics.get('base_model', 'bert-tiny')}`",
            f"- Eval accuracy: **{metrics.get('accuracy', 0):.3f}**, macro F1: **{metrics.get('macro_f1', 0):.3f}**",
            f"- Train/eval size: {metrics.get('train_size')}/{metrics.get('eval_size')}",
            "",
        ]

    lines += [
        "## What people are complaining about",
        "",
        "| # | Theme | Reviews | Likely surface | Top terms |",
        "|---|---|---:|---|---|",
    ]
    for i, theme in enumerate(themes, 1):
        terms = ", ".join(theme.get("top_terms", [])[:6]) or "—"
        surface = infer_surface(theme.get("top_terms", []))
        lines.append(f"| {i} | {theme_name(theme.get('top_terms', []))} | {theme.get('size', 0)} | {surface} | {terms} |")
    lines.append("")

    for i, theme in enumerate(themes, 1):
        name = theme_name(theme.get("top_terms", []))
        surface = infer_surface(theme.get("top_terms", []))
        heading = f"### Theme {i} — {name}" + (f" · {surface}" if surface != name else "")
        lines += [
            heading,
            "",
            f"**Evidence ({theme.get('size', 0)} reviews):**",
        ]
        for quote in theme.get("examples", [])[:3]:
            lines.append(f"> {quote[:300]}")
        lines += [
            "",
            f"**Suggested experiment:** {SURFACE_EXPERIMENTS.get(infer_surface(theme.get('top_terms', [])), SURFACE_EXPERIMENTS['Core workflow'])}",
            "",
        ]

    lines += [
        "## Method",
        "",
        "1. Public reviews are labelled from their star rating (1–2 negative, 3 neutral, 4–5 positive).",
        "2. A small transformer (`prajjwal1/bert-tiny`) is fine-tuned on a balanced public sample.",
        "3. Negative reviews are clustered with TF-IDF + KMeans; each cluster is represented by top terms and nearest-neighbour quotes.",
        "4. The report is generated from those clusters — no private data and no proprietary source code.",
        "",
        "## Limitations",
        "",
        "- Star ratings are a noisy proxy for sentiment; review text can be sarcastic or mixed.",
        "- The model is a lightweight baseline, not a production classifier. Treat the scores as directional.",
        "- Clusters are descriptive, not causal. The suggested experiments are hypotheses to validate.",
        "- Public reviews over-represent highly motivated users; pair this with support tickets and product analytics before deciding.",
        "",
    ]
    return "\n".join(lines)
