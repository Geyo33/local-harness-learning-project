import sqlite3
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import server  # noqa: F401


def test_server_importable():
    assert server._server is not None


def _make_db() -> tuple[sqlite3.Connection, str]:
    """Create a temp DB with the main app schema (episodes + key_facts + FTS5)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_name = tmp.name
    tmp.close()
    conn = sqlite3.connect(tmp_name)
    conn.execute("""
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            summary TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE key_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            source TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            summary, content='episodes', content_rowid='id'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts(rowid, summary) VALUES (new.id, new.summary);
        END
    """)
    conn.commit()
    return conn, tmp_name


def test_init_db_creates_episode_embeddings_table():
    conn, _ = _make_db()
    server.init_db(conn)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='episode_embeddings'"
    )
    assert cursor.fetchone() is not None


def test_init_db_sets_wal_mode():
    conn, _ = _make_db()
    server.init_db(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_init_db_cascade_delete():
    conn, _ = _make_db()
    conn.execute("PRAGMA foreign_keys = ON")
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", time.time(), "test summary", "agent"),
    )
    conn.commit()
    ep_id = conn.execute("SELECT id FROM episodes").fetchone()[0]
    vec = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
        (ep_id, vec),
    )
    conn.commit()
    conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
    conn.commit()
    row = conn.execute(
        "SELECT * FROM episode_embeddings WHERE episode_id = ?", (ep_id,)
    ).fetchone()
    assert row is None


def test_embed_texts_returns_vectors():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    with patch("httpx.post", return_value=mock_resp):
        result = server.embed_texts(["hello", "world"], "http://localhost:1234", "model")
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_texts_returns_none_on_error():
    with patch("httpx.post", side_effect=Exception("connection refused")):
        result = server.embed_texts(["hello"], "http://localhost:1234", "model")
    assert result is None


def test_semantic_search_ranks_by_cosine_similarity():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "python programming", "agent"),
    )
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 2000.0, "cooking recipes", "agent"),
    )
    conn.commit()

    # Store embeddings: ep1 = [1,0], ep2 = [0,1]
    ids = [r[0] for r in conn.execute("SELECT id FROM episodes ORDER BY id").fetchall()]
    conn.execute(
        "INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
        (ids[0], np.array([1.0, 0.0], dtype=np.float32).tobytes()),
    )
    conn.execute(
        "INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
        (ids[1], np.array([0.0, 1.0], dtype=np.float32).tobytes()),
    )
    conn.commit()

    # Query close to ep1
    results = server.semantic_search(conn, [0.9, 0.1], limit=2)
    assert results[0]["id"] == ids[0]
    assert results[0]["type"] == "episode"
    assert results[0]["score"] > results[1]["score"]


def test_semantic_search_returns_empty_when_no_embeddings():
    conn, _ = _make_db()
    server.init_db(conn)
    results = server.semantic_search(conn, [1.0, 0.0], limit=5)
    assert results == []


def test_semantic_search_handles_zero_norm_query():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "test", "agent"),
    )
    conn.commit()
    ep_id = conn.execute("SELECT id FROM episodes").fetchone()[0]
    conn.execute(
        "INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
        (ep_id, np.array([1.0, 0.0], dtype=np.float32).tobytes()),
    )
    conn.commit()
    results = server.semantic_search(conn, [0.0, 0.0], limit=5)
    assert results == []


def test_keyword_search_finds_episode_via_fts5():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "python programming tutorial", "agent"),
    )
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 2000.0, "cooking recipes book", "agent"),
    )
    conn.commit()

    results = server.keyword_search(conn, "python", limit=10)
    texts = [r["text"] for r in results]
    assert any("python" in t for t in texts)
    assert all(r["type"] == "episode" for r in results if r["text"] != "cooking recipes book")


def test_keyword_search_finds_key_fact_via_like():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO key_facts (fact, created_at, last_accessed, source) VALUES (?, ?, ?, ?)",
        ("user prefers dark mode", 1000.0, 1000.0, "agent"),
    )
    conn.commit()

    results = server.keyword_search(conn, "dark mode", limit=10)
    fact_results = [r for r in results if r["type"] == "fact"]
    assert len(fact_results) == 1
    assert "dark mode" in fact_results[0]["text"]


def test_keyword_search_normalizes_scores():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "python python python", "agent"),
    )
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 2000.0, "python programming", "agent"),
    )
    conn.commit()

    results = server.keyword_search(conn, "python", limit=10)
    ep_results = [r for r in results if r["type"] == "episode"]
    assert all(0.0 <= r["score"] <= 1.0 for r in ep_results)


def test_keyword_search_single_result_has_score_half():
    conn, _ = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "only match here", "agent"),
    )
    conn.commit()

    results = server.keyword_search(conn, "only match", limit=10)
    ep_results = [r for r in results if r["type"] == "episode"]
    assert len(ep_results) == 1
    assert ep_results[0]["score"] == 0.5


def _make_result(type_="episode", id_=1, text="test", created_at=1000.0, source="agent", score=0.5):
    return {"type": type_, "id": id_, "text": text, "created_at": created_at, "source": source, "score": score}


def test_entity_boost_raises_score_for_matching_entity():
    nlp_mock = MagicMock()
    ent = MagicMock()
    ent.text = "Python"
    chunk = MagicMock()
    chunk.text = "programming"
    doc = MagicMock()
    doc.ents = [ent]
    doc.noun_chunks = [chunk]
    nlp_mock.return_value = doc

    results = [_make_result(text="Python programming tutorial", score=0.5)]
    boosted = server.entity_boost(results, "Python programming", nlp_mock)
    assert boosted[0]["score"] > 0.5


def test_entity_boost_unchanged_when_no_match():
    nlp_mock = MagicMock()
    ent = MagicMock()
    ent.text = "JavaScript"
    doc = MagicMock()
    doc.ents = [ent]
    doc.noun_chunks = []
    nlp_mock.return_value = doc

    results = [_make_result(text="Python tutorial", score=0.5)]
    boosted = server.entity_boost(results, "JavaScript", nlp_mock)
    assert boosted[0]["score"] == 0.5


def test_fuse_results_deduplicates():
    group1 = [_make_result(id_=1, score=0.8)]
    group2 = [_make_result(id_=1, score=0.6)]
    fused = server.fuse_results([group1, group2], limit=10)
    assert len(fused) == 1
    assert fused[0]["score"] == pytest.approx(1.4)


def test_fuse_results_sorts_by_score_descending():
    group1 = [_make_result(id_=1, score=0.3), _make_result(id_=2, score=0.9)]
    fused = server.fuse_results([group1], limit=10)
    assert fused[0]["score"] > fused[1]["score"]


def test_fuse_results_respects_limit():
    group = [_make_result(id_=i, score=float(i)) for i in range(10)]
    fused = server.fuse_results([group], limit=3)
    assert len(fused) == 3


def test_format_results_empty():
    assert server.format_results([]) == "No memory matches found."


def test_format_results_formats_correctly():
    results = [_make_result(type_="episode", id_=1, text="test summary", created_at=1747612800.0, source="agent", score=1.0)]
    output = server.format_results(results)
    assert "[episode#1" in output
    assert "agent" in output
    assert "test summary" in output


def test_search_memory_handler_empty_query():
    conn, db_path = _make_db()
    server.init_db(conn)
    result = server.search_memory_handler("", 10, db_path, "http://localhost:1234", "model")
    assert result == "Error: 'query' is required."


def test_search_memory_handler_whitespace_query():
    conn, db_path = _make_db()
    server.init_db(conn)
    result = server.search_memory_handler("   ", 10, db_path, "http://localhost:1234", "model")
    assert result == "Error: 'query' is required."


def test_search_memory_handler_no_results():
    conn, db_path = _make_db()
    server.init_db(conn)
    with patch.object(server, "embed_texts", return_value=None):
        result = server.search_memory_handler("nothing here", 10, db_path, "http://localhost:1234", "model")
    assert result == "No memory matches found."


def test_search_memory_handler_returns_keyword_results_when_embedding_fails():
    conn, db_path = _make_db()
    server.init_db(conn)
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "python keyword match", "agent"),
    )
    conn.commit()
    with patch.object(server, "embed_texts", return_value=None):
        result = server.search_memory_handler("python", 10, db_path, "http://localhost:1234", "model")
    assert "python keyword match" in result


def test_search_memory_handler_deduplicates_semantic_and_keyword():
    conn, db_path = _make_db()
    server.init_db(conn)  # creates episode_embeddings table
    conn.execute(
        "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
        ("s1", 1000.0, "python tutorial", "agent"),
    )
    conn.commit()
    ep_id = conn.execute("SELECT id FROM episodes").fetchone()[0]
    conn.execute(
        "INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
        (ep_id, np.array([1.0, 0.0], dtype=np.float32).tobytes()),
    )
    conn.commit()

    with patch.object(server, "embed_texts", return_value=[[1.0, 0.0]]):
        result = server.search_memory_handler("python", 10, db_path, "http://localhost:1234", "model")

    # Should appear exactly once even though hit by both semantic and keyword
    assert result.count("python tutorial") == 1
