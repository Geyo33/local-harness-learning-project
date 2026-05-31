import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "memory_server"))
import server  # noqa: E402


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                summary TEXT NOT NULL,
                source TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS key_facts (
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
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, summary)
                    VALUES ('delete', old.id, old.summary);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, summary)
                    VALUES ('delete', old.id, old.summary);
                INSERT INTO episodes_fts(rowid, summary) VALUES (new.id, new.summary);
            END
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS key_facts_fts USING fts5(
                fact, content='key_facts', content_rowid='id'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS key_facts_ai AFTER INSERT ON key_facts BEGIN
                INSERT INTO key_facts_fts(rowid, fact) VALUES (new.id, new.fact);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS key_facts_ad AFTER DELETE ON key_facts BEGIN
                INSERT INTO key_facts_fts(key_facts_fts, rowid, fact)
                    VALUES ('delete', old.id, old.fact);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS key_facts_au AFTER UPDATE ON key_facts BEGIN
                INSERT INTO key_facts_fts(key_facts_fts, rowid, fact)
                    VALUES ('delete', old.id, old.fact);
                INSERT INTO key_facts_fts(rowid, fact) VALUES (new.id, new.fact);
            END
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episode_embeddings (
                episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )
        """)


def _insert_fact(conn: sqlite3.Connection, fact: str, source: str = "agent", days_ago: float = 10.0) -> int:
    ts = time.time() - days_ago * 86400
    with conn:
        cursor = conn.execute(
            "INSERT INTO key_facts (fact, created_at, last_accessed, source) VALUES (?, ?, ?, ?)",
            (fact, ts, ts, source),
        )
    return cursor.lastrowid


def test_keyword_search_uses_fts5_for_facts(tmp_path):
    db = tmp_path / "memory.db"
    with closing(sqlite3.connect(db)) as conn:
        _init_db(conn)
        _insert_fact(conn, "python project", "agent")
    with closing(sqlite3.connect(db)) as conn:
        results = server.keyword_search(conn, "python", 10)
    facts = [r for r in results if r["type"] == "fact"]
    assert len(facts) == 1
    assert facts[0]["text"] == "python project"
    assert 0.0 <= facts[0]["score"] <= 1.0


def test_keyword_search_fts5_score_normalized(tmp_path):
    db = tmp_path / "memory.db"
    with closing(sqlite3.connect(db)) as conn:
        _init_db(conn)
        _insert_fact(conn, "python python python", "agent")
        _insert_fact(conn, "python version info", "agent")
    with closing(sqlite3.connect(db)) as conn:
        results = server.keyword_search(conn, "python", 10)
    facts = [r for r in results if r["type"] == "fact"]
    assert len(facts) == 2
    scores = [r["score"] for r in facts]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] != scores[1]  # not flat 1.0


def test_keyword_search_fts5_fallback_on_syntax_error(tmp_path):
    db = tmp_path / "memory.db"
    with closing(sqlite3.connect(db)) as conn:
        _init_db(conn)
        _insert_fact(conn, "fact with bad( paren", "agent")
    with closing(sqlite3.connect(db)) as conn:
        # "bad(" triggers FTS5 OperationalError (unmatched paren);
        # LIKE '%bad(%' is a literal match and still finds the fact
        results = server.keyword_search(conn, "bad(", 10)
    facts = [r for r in results if r["type"] == "fact"]
    assert len(facts) == 1
    assert facts[0]["score"] == 1.0  # LIKE fallback gives flat score


def test_search_memory_handler_refreshes_last_accessed(tmp_path):
    db = tmp_path / "memory.db"
    with closing(sqlite3.connect(db)) as conn:
        _init_db(conn)
        old_ts = time.time() - 60 * 86400  # 60 days ago
        with conn:
            conn.execute(
                "INSERT INTO key_facts (fact, created_at, last_accessed, source) VALUES (?, ?, ?, ?)",
                ("python project", old_ts, old_ts, "agent"),
            )

    before = time.time()
    with patch("server.embed_texts", return_value=None):
        server.search_memory_handler("python", 10, str(db), "http://localhost:1234", "model")
    after = time.time()

    with closing(sqlite3.connect(db)) as conn:
        row = conn.execute(
            "SELECT last_accessed FROM key_facts WHERE fact = 'python project'"
        ).fetchone()
    assert before <= row[0] <= after


def test_search_memory_handler_no_refresh_when_no_facts(tmp_path):
    db = tmp_path / "memory.db"
    old_ts = time.time() - 60 * 86400
    with closing(sqlite3.connect(db)) as conn:
        _init_db(conn)
        with conn:
            conn.execute(
                "INSERT INTO key_facts (fact, created_at, last_accessed, source) VALUES (?, ?, ?, ?)",
                ("unrelated thing", old_ts, old_ts, "agent"),
            )
            conn.execute(
                "INSERT INTO episodes (session_id, created_at, summary, source) VALUES (?, ?, ?, ?)",
                ("sess-1", old_ts, "python session summary", "agent"),
            )

    with patch("server.embed_texts", return_value=None):
        server.search_memory_handler("python", 10, str(db), "http://localhost:1234", "model")

    with closing(sqlite3.connect(db)) as conn:
        row = conn.execute(
            "SELECT last_accessed FROM key_facts WHERE fact = 'unrelated thing'"
        ).fetchone()
    assert row[0] == old_ts  # unchanged — fact didn't match, not in results
