from complaintgraph.themes import cluster_complaints, rating_distribution


def test_clusters_have_terms_and_examples():
    texts = [
        "The app crashes every time I open the payment screen",
        "It freezes when I try to pay with my card",
        "Login keeps saying invalid password and I cannot reset it",
        "The OTP never arrives so I cannot sign in",
        "Search is slow and returns the wrong merchant",
        "Filtering transactions is confusing and laggy",
    ]
    themes = cluster_complaints(texts, n_clusters=3)
    assert 1 <= len(themes) <= 3
    assert sum(t["size"] for t in themes) == len(texts)
    assert all(t["top_terms"] for t in themes)
    assert all(t["examples"] for t in themes)


def test_rating_distribution():
    assert rating_distribution([1, 1, 5, 3]) == {"1": 2, "2": 0, "3": 1, "4": 0, "5": 1}
