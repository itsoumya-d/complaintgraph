"""Complaint theme discovery with TF-IDF + KMeans.

This is deliberately interpretable: every theme is a cluster with top terms
and evidence quotes, so a product team can argue with the output.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def rating_distribution(stars: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(s) for s in stars)
    return {str(i): counts.get(i, 0) for i in range(1, 6)}


def cluster_complaints(texts: list[str], n_clusters: int = 6, seed: int = 42) -> list[dict]:
    """Cluster complaint texts and return top terms plus representative quotes."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    n_clusters = max(1, min(n_clusters, len(texts)))

    domain_stop = [
        "app", "apps", "intercom", "please", "fix", "fixing", "doesn", "don", "won",
        "work", "working", "works", "new", "everytime", "really", "just", "like",
        "use", "using", "used", "get", "got", "one", "now", "still", "even",
    ]
    stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words() or []) + domain_stop
    min_df = 1 if len(texts) < 20 else 2
    max_df = 1.0 if len(texts) < 20 else 0.75
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=5000,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return [{
            "cluster": 0,
            "size": len(texts),
            "top_terms": [],
            "examples": texts[:3],
        }]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(matrix)
    terms = np.array(vectorizer.get_feature_names_out())
    distances = cosine_distances(matrix, kmeans.cluster_centers_)

    themes = []
    for cluster_id in range(n_clusters):
        member_idx = np.where(labels == cluster_id)[0]
        if len(member_idx) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        top_idx = centroid.argsort()[::-1][:8]
        top_terms = [str(terms[i]) for i in top_idx if centroid[i] > 0]
        order = member_idx[np.argsort(distances[member_idx, cluster_id])]
        examples = [texts[i] for i in order[:3]]
        themes.append({
            "cluster": int(cluster_id),
            "size": int(len(member_idx)),
            "top_terms": top_terms,
            "examples": examples,
        })

    themes.sort(key=lambda t: t["size"], reverse=True)
    return themes
