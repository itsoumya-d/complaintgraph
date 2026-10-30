from fastapi.testclient import TestClient

from complaintgraph.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_themes_endpoint():
    r = client.post("/themes", json={"reviews": [
        "It crashes when I pay",
        "The payment screen freezes",
        "Login OTP never arrives",
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["themes"]
    assert body["themes"][0]["size"] >= 1
