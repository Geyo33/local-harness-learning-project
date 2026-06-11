from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import logging
from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path

_FACT_SIMILARITY_THRESHOLD = 0.88
_FACT_JACCARD_THRESHOLD = 0.5
_PATTERN_SIMILARITY_THRESHOLD = 0.85


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

_CONFIDENCE_BY_SOURCE = {
    "task_success": 0.8,
    "user": 1.0,
    "agent": 0.6,
}


class EpisodicStore:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._agent_dir = db_path.parent
        self._init_db()
        self._migrate_playbook_json()

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
                        source        TEXT NOT NULL,
                        pinned        INTEGER NOT NULL DEFAULT 0,
                        tags          TEXT,
                        status        TEXT NOT NULL DEFAULT 'approved'
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
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS procedures (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern       TEXT NOT NULL,
                        action        TEXT NOT NULL,
                        confidence    REAL NOT NULL,
                        source        TEXT NOT NULL,
                        created_at    REAL NOT NULL,
                        last_accessed REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
                        pattern, action,
                        content='procedures',
                        content_rowid='id'
                    )
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
                        INSERT INTO procedures_fts(rowid, pattern, action)
                            VALUES (new.id, new.pattern, new.action);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
                        INSERT INTO procedures_fts(procedures_fts, rowid, pattern, action)
                            VALUES ('delete', old.id, old.pattern, old.action);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS procedures_au AFTER UPDATE ON procedures BEGIN
                        INSERT INTO procedures_fts(procedures_fts, rowid, pattern, action)
                            VALUES ('delete', old.id, old.pattern, old.action);
                        INSERT INTO procedures_fts(rowid, pattern, action)
                            VALUES (new.id, new.pattern, new.action);
                    END
                """)
                conn.execute("""
                    INSERT INTO procedures_fts(rowid, pattern, action)
                    SELECT p.id, p.pattern, p.action FROM procedures p
                    WHERE p.id NOT IN (SELECT rowid FROM procedures_fts)
                """)
        self._ensure_key_facts_columns()

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

    def remember_fact(self, fact: str, source: str = "agent", status: str = "approved") -> int:
        """Insert a key fact. Returns the new row id. status: 'approved' | 'pending'."""
        ts = time.time()
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO key_facts (fact, created_at, last_accessed, source, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (fact, ts, ts, source, status),
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

    def get_pending_facts(self) -> list[dict]:
        """Return all facts awaiting review, newest first."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, fact, created_at, source, tags FROM key_facts "
                "WHERE status = 'pending' ORDER BY created_at DESC, id DESC"
            )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "fact": r[1], "created_at": r[2], "source": r[3], "tags": r[4]}
            for r in rows
        ]

    def approve_fact(self, fact_id: int, new_fact: str | None = None, pinned: bool = False) -> bool:
        """Promote a pending fact to approved, optionally editing its text and pinning it.
        Returns True if a row was updated."""
        ts = time.time()
        pin_val = 1 if pinned else 0
        with closing(self._connect()) as conn:
            with conn:
                if new_fact is not None and new_fact.strip():
                    cursor = conn.execute(
                        "UPDATE key_facts SET fact = ?, status = 'approved', pinned = ?, "
                        "last_accessed = ? WHERE id = ?",
                        (new_fact.strip(), pin_val, ts, fact_id),
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE key_facts SET status = 'approved', pinned = ?, "
                        "last_accessed = ? WHERE id = ?",
                        (pin_val, ts, fact_id),
                    )
                return cursor.rowcount > 0

    def count_pending(self) -> int:
        """Return the number of facts with status='pending'."""
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM key_facts WHERE status = 'pending'"
            ).fetchone()[0]

    def get_key_facts(self, limit: int = 20) -> list[dict]:
        """Return approved key facts ranked by source-weighted time decay score."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, fact, created_at, source, last_accessed FROM key_facts "
                "WHERE status = 'approved' "
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

    def get_pinned_facts(self, limit: int = 20) -> list[dict]:
        """Return approved + pinned facts ranked by source-weighted time decay."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, fact, created_at, source, last_accessed FROM key_facts "
                "WHERE status = 'approved' AND pinned = 1 "
                "LIMIT ?",
                (_MAX_KEY_FACTS_FETCH,),
            )
            rows = cursor.fetchall()
        now = time.time()
        scored = []
        for r in rows:
            days = (now - r[4]) / 86400.0
            score = _SOURCE_WEIGHT.get(r[3], 0.8) * math.exp(-_DECAY_LAMBDA * days)
            scored.append((score, {"id": r[0], "fact": r[1], "created_at": r[2], "source": r[3]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def search_memory(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 BM25 search on episodes + key_facts.

        Query tokens are OR-joined (each quoted as a literal) so a conversational
        request matches rows sharing only some keywords; FTS5's default implicit-
        AND would otherwise require every filler word to appear and match nothing
        (mirrors search_facts / search_procedures). The LIKE fallbacks still use
        the raw substring."""
        tokens = re.findall(r"\w+", query.lower())
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        results: list[dict] = []
        with closing(self._connect()) as conn:
            try:
                if not fts_query:
                    raise sqlite3.OperationalError("empty query")
                cursor = conn.execute(
                    """
                    SELECT 'episode' AS type, e.id, e.summary, e.created_at, e.source, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.rowid
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
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
                if not fts_query:
                    raise sqlite3.OperationalError("empty query")
                cursor = conn.execute(
                    """
                    SELECT 'fact' AS type, kf.id, kf.fact, kf.created_at, kf.source, f.rank
                    FROM key_facts_fts f
                    JOIN key_facts kf ON kf.id = f.rowid
                    WHERE key_facts_fts MATCH ? AND kf.status = 'approved'
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": row[0], "id": row[1], "text": row[2],
                        "created_at": row[3], "source": row[4], "score": row[5],
                    })
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "SELECT id, fact, created_at, source FROM key_facts "
                    "WHERE fact LIKE ? AND status = 'approved' ORDER BY last_accessed DESC LIMIT ?",
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

    def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        """Facts-only FTS5 BM25 search (approved facts). Separate from
        search_memory so the ephemeral fact window isn't crowded out: episodes
        and facts share search_memory's single merged `limit` cut (and their
        BM25 ranks aren't cross-comparable), so episode hits can evict every
        fact before the type filter runs. This path queries only key_facts.

        Query tokens are OR-joined (each quoted as a literal): a fact matches if
        it contains *any* token, with BM25 still ranking denser matches higher.
        Without this, FTS5's default implicit-AND requires every token — including
        filler words from a conversational query — to appear, so a natural-
        language question matches nothing (mirrors search_procedures)."""
        tokens = re.findall(r"\w+", query.lower())
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        results: list[dict] = []
        with closing(self._connect()) as conn:
            try:
                if not fts_query:
                    raise sqlite3.OperationalError("empty query")
                cursor = conn.execute(
                    """
                    SELECT 'fact' AS type, kf.id, kf.fact, kf.created_at, kf.source, f.rank
                    FROM key_facts_fts f
                    JOIN key_facts kf ON kf.id = f.rowid
                    WHERE key_facts_fts MATCH ? AND kf.status = 'approved'
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": row[0], "id": row[1], "text": row[2],
                        "created_at": row[3], "source": row[4], "score": row[5],
                    })
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "SELECT id, fact, created_at, source FROM key_facts "
                    "WHERE fact LIKE ? AND status = 'approved' ORDER BY last_accessed DESC LIMIT ?",
                    (f"%{query}%", limit),
                )
                for row in cursor.fetchall():
                    results.append({
                        "type": "fact", "id": row[0], "text": row[1],
                        "created_at": row[2], "source": row[3], "score": 0,
                    })

            fact_ids = [r["id"] for r in results]
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

    def record_procedure(self, pattern: str, action: str, source: str = "task_success") -> int:
        """Insert or dedup-bump a procedure. Returns the row id."""
        ts = time.time()
        with closing(self._connect()) as conn:
            cursor = conn.execute("SELECT id, pattern FROM procedures")
            rows = cursor.fetchall()
        for row_id, existing_pattern in rows:
            if SequenceMatcher(None, pattern.lower(), existing_pattern.lower()).ratio() >= _PATTERN_SIMILARITY_THRESHOLD:
                with closing(self._connect()) as conn:
                    with conn:
                        conn.execute(
                            "UPDATE procedures SET confidence = ROUND(MIN(1.0, confidence + 0.05), 2), "
                            "action = ?, last_accessed = ? WHERE id = ?",
                            (action, ts, row_id),
                        )
                return row_id
        confidence = _CONFIDENCE_BY_SOURCE.get(source, 0.6)
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO procedures (pattern, action, confidence, source, created_at, last_accessed) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (pattern, action, confidence, source, ts, ts),
                )
                return cursor.lastrowid

    def get_top_procedures(self, n: int) -> list[dict]:
        """Return top-n procedures ranked by decay score."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, pattern, action, confidence, source, created_at, last_accessed FROM procedures"
            )
            rows = cursor.fetchall()
        now = time.time()
        scored = []
        for r in rows:
            days = (now - r[6]) / 86400.0
            score = r[3] * math.exp(-_DECAY_LAMBDA * days)
            scored.append((score, {
                "id": r[0], "pattern": r[1], "action": r[2],
                "confidence": r[3], "source": r[4], "created_at": r[5],
            }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]

    def search_procedures(self, query: str, limit: int = 5) -> list[dict]:
        """FTS5 BM25 search on procedures, re-ranked by confidence + disuse decay.

        Query tokens are OR-joined: a procedure matches if it contains *any*
        token (BM25 still ranks rows matching more tokens higher). This keeps
        recall usable when the caller passes a full natural-language request
        (e.g. the planner injection), where FTS5's default implicit-AND would
        otherwise require every filler word to appear and match nothing.

        Final ranking blends keyword relevance with the same confidence + decay
        score as ``get_top_procedures`` / ``get_key_facts``
        (``confidence * exp(-_DECAY_LAMBDA * days_since_last_accessed)``). This
        is what lets the learning loop work: a procedure penalized by
        ``adjust_confidence`` sinks in recall before it is pruned at the floor.
        FTS gives a candidate pool; the decay-weighted score picks the winners.
        """
        # Quote each alnum token so FTS5 treats it as a literal, not an operator.
        tokens = re.findall(r"\w+", query.lower())
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        # Pull a wider candidate pool than `limit` so the decay re-rank has room
        # to demote stale/low-confidence matches below fresher, weaker-keyword ones.
        pool = max(limit * 5, 25)
        now = time.time()
        results: list[dict] = []
        with closing(self._connect()) as conn:
            candidates: list[tuple[float, dict]] = []
            try:
                if not fts_query:
                    raise sqlite3.OperationalError("empty query")
                # Weight the pattern column above action: the query resembles a
                # trigger (pattern) more than a recipe (action), so a match in
                # the trigger is a stronger relevance signal than one buried in
                # the steps. Lower bm25 = better, hence ORDER BY ascending.
                cursor = conn.execute(
                    "SELECT p.id, p.pattern, p.action, p.confidence, p.source, p.created_at, "
                    "p.last_accessed, bm25(procedures_fts, 5.0, 1.0) AS score "
                    "FROM procedures_fts f JOIN procedures p ON p.id = f.rowid "
                    "WHERE procedures_fts MATCH ? ORDER BY score LIMIT ?",
                    (fts_query, pool),
                )
                rows = cursor.fetchall()
                # Normalize bm25 (lower=better, often negative) to relevance in
                # (0, 1]: the best match in the pool scores 1.0, others taper off.
                best = rows[0][7] if rows else 0.0
                for row in rows:
                    rel = 1.0 / (1.0 + (row[7] - best))
                    decay = row[3] * math.exp(-_DECAY_LAMBDA * (now - row[6]) / 86400.0)
                    candidates.append((rel * decay, {
                        "id": row[0], "pattern": row[1], "action": row[2],
                        "confidence": row[3], "source": row[4], "created_at": row[5],
                    }))
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "SELECT id, pattern, action, confidence, source, created_at, last_accessed "
                    "FROM procedures WHERE pattern LIKE ? OR action LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", pool),
                )
                for row in cursor.fetchall():
                    # No keyword-relevance signal in the LIKE fallback — rank on
                    # confidence + decay alone.
                    decay = row[3] * math.exp(-_DECAY_LAMBDA * (now - row[6]) / 86400.0)
                    candidates.append((decay, {
                        "id": row[0], "pattern": row[1], "action": row[2],
                        "confidence": row[3], "source": row[4], "created_at": row[5],
                    }))
            candidates.sort(key=lambda x: x[0], reverse=True)
            results = [item for _, item in candidates[:limit]]
            proc_ids = [r["id"] for r in results]
            if proc_ids:
                try:
                    with conn:
                        conn.executemany(
                            "UPDATE procedures SET last_accessed = ? WHERE id = ?",
                            [(now, pid) for pid in proc_ids],
                        )
                except sqlite3.DatabaseError:
                    logging.warning("Failed to refresh last_accessed for procedure ids %s", proc_ids)
        return results

    def adjust_confidence(self, ids: list[int], delta: float, floor: float = 0.1) -> None:
        """Adjust confidence by delta for given ids; prune rows at or below floor."""
        if not ids:
            return
        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    "UPDATE procedures SET confidence = ROUND(MIN(1.0, MAX(0.0, confidence + ?)), 2) WHERE id = ?",
                    [(delta, pid) for pid in ids],
                )
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM procedures WHERE id IN ({placeholders}) AND confidence <= ?",
                    [*ids, floor],
                )

    def _ensure_key_facts_columns(self) -> None:
        """Idempotently add pinned/tags/status columns to an existing key_facts table.

        Upgrade path for pre-existing databases; on a fresh DB the guards no-op.
        status domain: 'pending' (awaiting review) | 'approved' (default, visible to LLM).
        """
        with closing(self._connect()) as conn:
            with conn:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(key_facts)")}
                if "pinned" not in cols:
                    conn.execute(
                        "ALTER TABLE key_facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                    )
                if "tags" not in cols:
                    conn.execute("ALTER TABLE key_facts ADD COLUMN tags TEXT")
                if "status" not in cols:
                    conn.execute(
                        "ALTER TABLE key_facts ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'"
                    )

    def _migrate_playbook_json(self) -> None:
        """One-time import of legacy playbook.json into the procedures table."""
        legacy = self._agent_dir / "playbook.json"
        migrated = self._agent_dir / "playbook.json.migrated"
        if not legacy.exists() or migrated.exists():
            return
        with closing(self._connect()) as conn:
            count = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        if count > 0:
            legacy.rename(migrated)
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            procedures = data.get("procedures", [])
        except Exception as e:
            logging.warning("playbook.json migration: could not read legacy file: %s", e)
            legacy.rename(migrated)  # don't retry a corrupt file every startup
            return
        imported = 0
        if procedures:
            with closing(self._connect()) as conn:
                with conn:
                    for p in procedures:
                        try:
                            ts = p.get("created_at", time.time())
                            conn.execute(
                                "INSERT INTO procedures "
                                "(pattern, action, confidence, source, created_at, last_accessed) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (p["pattern"], p["action"], p.get("confidence", 0.8),
                                 p.get("source", "task_success"), ts, ts),
                            )
                            imported += 1
                        except (KeyError, TypeError) as e:
                            logging.warning("playbook.json migration: skipping malformed entry: %s", e)
            logging.info("Migrated %d procedures from playbook.json to memory.db", imported)
        legacy.rename(migrated)

    def clear_procedures(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM procedures")
        logging.info("EpisodicStore: cleared all procedures")

    def clear_all(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM episodes")
                conn.execute("DELETE FROM key_facts")
        logging.info("EpisodicStore: cleared all episodes and key_facts")
