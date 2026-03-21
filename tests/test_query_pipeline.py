import pytest


def test_keyword_parser_returns_dict():
    try:
        from megarag.retrieval.keyword_parser import parse_keywords
    except Exception:
        pytest.skip("Azure OpenAI not configured")

    result = parse_keywords("What are the revenue figures for Q3?")
    assert "low_level" in result
    assert "high_level" in result
    assert isinstance(result["low_level"], list)
