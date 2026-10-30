from complaintgraph.report import build_report, infer_surface


def test_infer_surface():
    assert infer_surface(["crash", "freeze"]) == "Reliability / performance"
    assert infer_surface(["refund", "billing"]) == "Billing / payments"
    assert infer_surface(["something", "else"]) == "Core workflow"


def test_report_sections():
    rows = [
        {"text": "It crashes on payment", "star": 1, "label": 0},
        {"text": "Login OTP never arrives", "star": 2, "label": 0},
        {"text": "Great app", "star": 5, "label": 2},
    ]
    themes = [{"cluster": 0, "size": 2, "top_terms": ["crash", "payment"], "examples": ["It crashes on payment"]}]
    md = build_report("Demo App", rows, themes, metrics={"accuracy": 0.7, "macro_f1": 0.68, "train_size": 10, "eval_size": 2})
    assert "# ComplaintGraph — Demo App" in md
    assert "What people are complaining about" in md
    assert "Theme 1" in md
    assert "Reliability / performance" in md
    assert "Limitations" in md
