import pytest
from unittest.mock import MagicMock


def _bare_session():
    from mcp_chatbot.core.session import ChatSession
    s = ChatSession.__new__(ChatSession)
    s.episodic_store = MagicMock()
    s._fact_window = []
    s._retrieval_k = 5
    s._retrieval_ttl = 3
    return s


def test_update_fact_window_adds_hits():
    s = _bare_session()
    # search_facts is facts-only; the episode here is a defensive case the
    # type filter still drops.
    s.episodic_store.search_facts.return_value = [
        {"type": "fact", "id": 1, "text": "user likes dark mode", "created_at": 0, "source": "user", "score": -1},
        {"type": "episode", "id": 9, "text": "ignored episode", "created_at": 0, "source": "agent", "score": -1},
    ]
    s._update_fact_window("dark mode")
    ids = [f["id"] for f in s._fact_window]
    assert ids == [1]  # episodes excluded


def test_fact_window_ttl_expires():
    s = _bare_session()
    s.episodic_store.search_facts.return_value = [
        {"type": "fact", "id": 1, "text": "ephemeral fact", "created_at": 0, "source": "user", "score": -1},
    ]
    s._update_fact_window("q")          # ttl reset to 3
    s.episodic_store.search_facts.return_value = []
    s._update_fact_window("other")      # 3 -> 2
    s._update_fact_window("other")      # 2 -> 1
    s._update_fact_window("other")      # 1 -> 0, dropped
    assert s._fact_window == []


def test_render_fact_window_block():
    s = _bare_session()
    s._fact_window = [{"id": 1, "fact": "alpha", "ttl": 3}]
    block = s._render_fact_window()
    assert "alpha" in block
    assert "[Relevant Facts]" in block
