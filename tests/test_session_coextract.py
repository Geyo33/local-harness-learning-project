import json
from unittest.mock import MagicMock


def _coextract_session():
    from mcp_chatbot.core.session import ChatSession
    s = ChatSession.__new__(ChatSession)
    s.llm_client = MagicMock()
    return s


def test_summarize_with_facts_parses_json():
    s = _coextract_session()
    s.llm_client.get_response.return_value = {
        "content": json.dumps({"summary": "did stuff", "facts": ["fact one", "fact two"]})
    }
    summary, facts = s._summarize_with_facts([{"role": "user", "content": "hi"}])
    assert summary == "did stuff"
    assert facts == ["fact one", "fact two"]
    _, kwargs = s.llm_client.get_response.call_args
    assert "response_format" in kwargs


def test_summarize_with_facts_falls_back_on_bad_json():
    s = _coextract_session()
    s.llm_client.get_response.return_value = {"content": "plain text summary, no json"}
    summary, facts = s._summarize_with_facts([{"role": "user", "content": "hi"}])
    assert summary == "plain text summary, no json"
    assert facts == []


def test_summarize_with_facts_empty_content():
    s = _coextract_session()
    s.llm_client.get_response.return_value = {"content": ""}
    summary, facts = s._summarize_with_facts([{"role": "user", "content": "hi"}])
    assert summary is None
    assert facts == []
