from __future__ import annotations

import math
import sqlite3
import time
import logging
from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path

_FACT_SIMILARITY_THRESHOLD = 0.88
_FACT_JACCARD_THRESHOLD = 0.5


def _fact_similar(a: str, b: str) -> bool:
    if SequenceMatcher(None, a.lower(), b.lower()).ratio() < _FACT_SIMILARITY_THRESHOLD:
        return False
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    jaccard = len(sa & sb) / len(union) if union else 0.0
    return jaccard >= _FACT_JACCARD_THRESHOLD


# unknown source → fallback 0.8 (slightly above agent, treated as mildly trusted)
_SOURCE_WEIGHT = {"user": 1.0, "agent": 0.7}  # decay weight, distinct from playbook's confidence scale
_DECAY_LAMBDA = 0.01  # half-life ~70 days
_MAX_KEY_FACTS_FETCH = 500


class EpisodicStore:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT    NOT NULL,
                        created_at REAL    NOT NULL,
                        summary    TEXT    NOT NULL,
                        source     TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS key_facts (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        fact          TEXT NOT NULL,
                        created_at    REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        source        TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_created_at
                    ON episodes(created_at DESC)
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                        summary,
                        content='episodes',
                        content_rowid='id'
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
                        fact,
                        content='key_facts',
                        content_rowid='id'
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
                    INSERT INTO episodes_fts(rowid, summary)
                    SELECT e.id, e.summary FROM episodes e
                    WHERE e.id NOT IN (SELECT rowid FROM episodes_fts)
                """)
                conn.execute("""
                    INSERT INTO key_facts_fts(rowid, fact)
                    SELECT kf.id, kf.fact FROM key_facts kf
                    WHERE kf.id NOT IN (SELECT rowid FROM key_facts_fts)
                """)

    def add_episode(
        self,
        session_id: str,
        summary: str,
        source: str,
        created_at: float | None = None,
    ) -> None:
        ts = created_at if created_at is not None else time.time()
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO episodes (session_id, created_at, summary, source) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, ts, summary, source),
                )
        logging.debug("EpisodicStore: wrote episode for session %s", session_id)

    def get_recent(self, n: int) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT session_id, created_at, summary, source "
                "FROM episodes ORDER BY created_at DESC, id DESC LIMIT ?",
                (n,),
            )
            rows = cursor.fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "summary": r[2], "source": r[3]}
            for r in rows
        ]

    def remember_fact(self, fact: str, source: str = "agent") -> int:
        """Insert a key fact. Returns the new row id."""
        ts = time.time()
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO key_facts (fact, created_at, last_accessed, source) "
                    "VALUES (?, ?, ?, ?)",
                    (fact, ts, ts, source),
                )
                return cursor.lastrowid

    def find_similar_fact(self, text: str) -> dict | None:
        """Full scan of key_facts; return first row similar to text, or None."""
        with closing(self._connect()) as conn:
            cursor = conn.execute("SELECT id, fact FROM key_facts")
            for row in cursor.fetchall():
                if _fact_similar(row[1], text):
                    return {"id": row[0], "fact": row[1]}
        return None

    def update_fact(self, fact_id: int, new_fact: str) -> bool:
        """Update an existing key fact. Returns True if a row was updated."""
        ts = time.time()
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE key_facts SET fact = ?, last_accessed = ? WHERE id = ?",
                    (new_fact, ts, fact_id),
                )
                return cursor.rowcount > 0

    def forget_fact(self, fact_id: int) -> bool:
        """Delete a key fact by id. Returns True if a row was deleted."""
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM key_facts WHERE id = ?", (fact_id,)
                )
                return cursor.rowcount > 0

    def get_key_facts(self, limit: int = 20) -> list[dict]:
        """Return key facts ranked by source-weighted time decay score."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, fact, created_at, source, last_accessed FROM key_facts "
                "LIMIT ?",
                (_MAX_KEY_FACTS_FETCH,),
            )
            rows = cursor.fetchall()
        if len(rows) == _MAX_KEY_FACTS_FETCH:
            logging.warning("get_key_facts: fetch capped at %d rows", _MAX_KEY_FACTS_FETCH)
        now = time.time()
        scored = []
        for r in rows:
            days = (now - r[4]) / 86400.0
            score = _SOURCE_WEIGHT.get(r[3], 0.8) * math.exp(-_DECAY_LAMBDA * days)
            scored.append((score, {"id": r[0], "fact": r[1], "created_at": r[2], "source": r[3]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def search_memory(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 BM25 search on episodes + key_facts."""
        results: list[dict] = []
        with closing(self._connect()) as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT 'episode' AS type, e.id, e.summary, e.created_at, e.source, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.rowid
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": row[0], "id": row[1], "text": row[2],
                        "created_at": row[3], "source": row[4], "score": row[5],
                    })
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "SELECT id, summary, created_at, source FROM episodes "
                    "WHERE summary LIKE ? LIMIT ?",
                    (f"%{query}%", limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": "episode", "id": row[0], "text": row[1],
                        "created_at": row[2], "source": row[3], "score": 0,
                    })

            try:
                cursor = conn.execute(
                    """
                    SELECT 'fact' AS type, kf.id, kf.fact, kf.created_at, kf.source, f.rank
                    FROM key_facts_fts f
                    JOIN key_facts kf ON kf.id = f.rowid
                    WHERE key_facts_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": row[0], "id": row[1], "text": row[2],
                        "created_at": row[3], "source": row[4], "score": row[5],
                    })
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "SELECT id, fact, created_at, source FROM key_facts "
                    "WHERE fact LIKE ? ORDER BY last_accessed DESC LIMIT ?",
                    (f"%{query}%", limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": "fact", "id": row[0], "text": row[1],
                        "created_at": row[2], "source": row[3], "score": 0,
                    })

            results.sort(key=lambda r: r["score"])
            results = results[:limit]

            fact_ids = [r["id"] for r in results if r["type"] == "fact"]
            if fact_ids:
                try:
                    now = time.time()
                    with conn:
                        conn.executemany(
                            "UPDATE key_facts SET last_accessed = ? WHERE id = ?",
                            [(now, fid) for fid in fact_ids],
                        )
                except sqlite3.DatabaseError:
                    logging.warning(
                        "Failed to refresh last_accessed for fact ids %s", fact_ids
                    )

        return results

    def clear_all(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM episodes")
                conn.execute("DELETE FROM key_facts")
        logging.info("EpisodicStore: cleared all episodes and key_facts")
