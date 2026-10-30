from complaintgraph.labels import clean_text, englishish, star_to_label


def test_star_mapping():
    assert star_to_label(1) == 0
    assert star_to_label(2) == 0
    assert star_to_label(3) == 1
    assert star_to_label(4) == 2
    assert star_to_label(5) == 2


def test_clean_text_removes_urls_and_html():
    text = "  <b>Love it</b> https://example.com  but crashes  "
    cleaned = clean_text(text)
    assert "example.com" not in cleaned
    assert "<b>" not in cleaned
    assert "crashes" in cleaned


def test_englishish_filters_non_ascii():
    assert englishish("This app crashes every time I open it.")
    assert not englishish("这个应用每次打开都崩溃" * 5)
