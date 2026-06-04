from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import httpx
import numpy as np
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
import mcp.types as types

logging.basicConfig(level=logging.INFO)

# ── MCP server instance ───────────────────────────────────────────────────────

_server = Server("memory-retrieval")
_config: dict = {}
_db_initialized: bool = False


@_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_memory",
            description=(
                "Search long-term memory for episodes and facts matching a query. "
                "Uses semantic, keyword, and entity signals. "
                "Returns ranked results. Use before answering questions about past sessions or stored knowledge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or phrase to search for."},
                    "limit": {"type": "integer", "description": "Max results to return.", "default": 10},
                },
                "required": ["query"],
            },
        )
    ]


@_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    arguments = arguments or {}
    if name == "search_memory":
        query = arguments.get("query", "").strip()
        limit = int(arguments.get("limit", 10))
        result = await asyncio.to_thread(
            search_memory_handler,
            query,
            limit,
            _config["db_path"],
            _config["embed_base_url"],
            _config["embed_model"],
            _config.get("nlp"),
        )
        return [types.TextContent(type="text", text=result)]
    return [types.TextContent(type="text", text=f"Error: unknown tool '{name}'.")]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episode_embeddings (
            episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
            embedding  BLOB NOT NULL
        )
    """)
    conn.commit()

def embed_texts(texts: list[str], base_url: str, model: str) -> list[list[float]] | None:
    try:
        resp = httpx.post(
            f"{base_url}/v1/embeddings",
            json={"input": texts, "model": model},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]
    except Exception as e:
        logging.warning("Embedding call failed: %s", e)
        return None

def semantic_search(conn: sqlite3.Connection, query_vec: list[float], limit: int) -> list[dict]:
    cursor = conn.execute(
        "SELECT e.id, e.summary, e.created_at, e.source, ee.embedding "
        "FROM episode_embeddings ee JOIN episodes e ON e.id = ee.episode_id"
    )
    rows = cursor.fetchall()
    if not rows:
        return []
    q = np.array(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    results = []
    for row in rows:
        vec = np.frombuffer(row[4], dtype=np.float32)
        v_norm = float(np.linalg.norm(vec))
        score = float(np.dot(q, vec) / (q_norm * v_norm)) if v_norm > 0 else 0.0
        results.append({
            "type": "episode", "id": row[0], "text": row[1],
            "created_at": row[2], "source": row[3], "score": score,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def keyword_search(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    results = []

    # FTS5 on episodes
    try:
        cursor = conn.execute(
            "SELECT e.id, e.summary, e.created_at, e.source, f.rank "
            "FROM episodes_fts f JOIN episodes e ON e.id = f.rowid "
            "WHERE episodes_fts MATCH ? ORDER BY f.rank LIMIT ?",
            (query, limit),
        )
        rows = cursor.fetchall()
        if rows:
            ranks = [r[4] for r in rows]
            min_r, max_r = min(ranks), max(ranks)
            rng = max_r - min_r
            for r in rows:
                score = (max_r - r[4]) / rng if rng != 0.0 else 0.5
                results.append({
                    "type": "episode", "id": r[0], "text": r[1],
                    "created_at": r[2], "source": r[3], "score": score,
                })
    except sqlite3.OperationalError as e:
        logging.warning("episodes_fts unavailable, falling back to LIKE search: %s", e)
        cursor = conn.execute(
            "SELECT id, summary, created_at, source FROM episodes "
            "WHERE summary LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
        for r in cursor.fetchall():
            results.append({
                "type": "episode", "id": r[0], "text": r[1],
                "created_at": r[2], "source": r[3], "score": 1.0,
            })

    # FTS5 on key_facts
    try:
        cursor = conn.execute(
            "SELECT kf.id, kf.fact, kf.created_at, kf.source, f.rank "
            "FROM key_facts_fts f JOIN key_facts kf ON kf.id = f.rowid "
            "WHERE key_facts_fts MATCH ? ORDER BY f.rank LIMIT ?",
            (query, limit),
        )
        rows = cursor.fetchall()
        if rows:
            ranks = [r[4] for r in rows]
            min_r, max_r = min(ranks), max(ranks)
            rng = max_r - min_r
            for r in rows:
                score = (max_r - r[4]) / rng if rng != 0.0 else 0.5
                results.append({
                    "type": "fact", "id": r[0], "text": r[1],
                    "created_at": r[2], "source": r[3], "score": score,
                })
    except sqlite3.OperationalError as e:
        logging.warning("key_facts_fts unavailable, falling back to LIKE search: %s", e)
        cursor = conn.execute(
            "SELECT id, fact, created_at, source FROM key_facts "
            "WHERE fact LIKE ? ORDER BY last_accessed DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        for r in cursor.fetchall():
            results.append({
                "type": "fact", "id": r[0], "text": r[1],
                "created_at": r[2], "source": r[3], "score": 1.0,
            })

    # FTS5 on procedures
    try:
        cursor = conn.execute(
            "SELECT p.id, p.pattern, p.action, p.created_at, p.source, f.rank "
            "FROM procedures_fts f JOIN procedures p ON p.id = f.rowid "
            "WHERE procedures_fts MATCH ? ORDER BY f.rank LIMIT ?",
            (query, limit),
        )
        rows = cursor.fetchall()
        if rows:
            ranks = [r[5] for r in rows]
            min_r, max_r = min(ranks), max(ranks)
            rng = max_r - min_r
            for r in rows:
                score = (max_r - r[5]) / rng if rng != 0.0 else 0.5
                results.append({
                    "type": "procedure", "id": r[0], "text": f"{r[1]} → {r[2]}",
                    "created_at": r[3], "source": r[4], "score": score,
                })
    except sqlite3.OperationalError as e:
        logging.warning("procedures_fts unavailable, falling back to LIKE search: %s", e)
        try:
            cursor = conn.execute(
                "SELECT id, pattern, action, created_at, source FROM procedures "
                "WHERE pattern LIKE ? OR action LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
            for r in cursor.fetchall():
                results.append({
                    "type": "procedure", "id": r[0], "text": f"{r[1]} → {r[2]}",
                    "created_at": r[3], "source": r[4], "score": 1.0,
                })
        except sqlite3.OperationalError:
            pass  # procedures table not yet available on this DB

    return results

def entity_boost(results: list[dict], query: str, nlp) -> list[dict]:
    doc = nlp(query)
    entities = (
        {ent.text.lower() for ent in doc.ents}
        | {chunk.text.lower() for chunk in doc.noun_chunks}
    )
    if not entities:
        return results
    return [
        {**r, "score": r["score"] + sum(0.2 for e in entities if re.search(r'\b' + re.escape(e) + r'\b', r["text"].lower()))}
        for r in results
    ]

def fuse_results(result_groups: list[list[dict]], limit: int) -> list[dict]:
    fused: dict[tuple, dict] = {}
    for group in result_groups:
        for r in group:
            key = (r["type"], r["id"])
            if key in fused:
                fused[key]["score"] += r["score"]
            else:
                fused[key] = {**r}
    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:limit]

def format_results(results: list[dict], query: str) -> str:
    if not results:
        return f"No memory matches found. Nothing useful in memory for the query: '{query}'."
    lines = []
    for r in results:
        date_str = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
        tag = f"[{r['type']}#{r['id']} {date_str} {r['source']}]"
        lines.append(f"{tag} {r['text']}")
    return "\n".join(lines)

def search_memory_handler(
    query: str,
    limit: int,
    db_path: str,
    embed_base_url: str,
    embed_model: str,
    nlp=None,
) -> str:
    if not query or not query.strip():
        return "Error: 'query' is required."

    global _db_initialized
    with closing(sqlite3.connect(db_path)) as conn:
        if not _db_initialized:
            init_db(conn)
            _db_initialized = True

        result_groups: list[list[dict]] = []

        # Backfill missing embeddings and embed the query in a single batch call
        missing_rows = conn.execute(
            "SELECT id, summary FROM episodes "
            "WHERE id NOT IN (SELECT episode_id FROM episode_embeddings)"
        ).fetchall()
        texts_to_embed = [r[1] for r in missing_rows] + [query]
        all_vecs = embed_texts(texts_to_embed, embed_base_url, embed_model)
        if all_vecs is not None:
            # Write backfill embeddings
            backfill_vecs = all_vecs[:-1]
            query_vec = all_vecs[-1]
            if missing_rows:
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
                        [
                            (missing_rows[i][0], np.array(backfill_vecs[i], dtype=np.float32).tobytes())
                            for i in range(len(missing_rows))
                        ],
                    )
            result_groups.append(semantic_search(conn, query_vec, limit))
        # else: embedding unavailable — skip semantic pass, keyword still runs

        # Pass 2: keyword
        result_groups.append(keyword_search(conn, query, limit))

        # Fuse passes 1 + 2
        fused = fuse_results(result_groups, limit * 3)

        # Pass 3: entity boost (post-fusion)
        if nlp is not None:
            fused = entity_boost(fused, query, nlp)

        fused.sort(key=lambda x: x["score"], reverse=True)
        final_results = fused[:limit]

        fact_ids = [r["id"] for r in final_results if r["type"] == "fact"]
        if fact_ids:
            now = time.time()
            try:
                with conn:
                    conn.executemany(
                        "UPDATE key_facts SET last_accessed = ? WHERE id = ?",
                        [(now, fid) for fid in fact_ids],
                    )
            except sqlite3.DatabaseError:
                logging.warning("Failed to refresh last_accessed for fact ids %s", fact_ids)

        return format_results(final_results, query)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory retrieval MCP server")
    parser.add_argument("--db-path", default=None, help="Path to memory.db (default: read from mcp_chatbot/settings.json)")
    parser.add_argument("--embed-url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--embed-model", default="text-embedding-bge-small-en-v1.5", help="Embedding model name")
    args = parser.parse_args()

    if args.db_path:
        db_path = args.db_path
    else:
        try:
            settings = json.loads((Path(__file__).parent.parent / "mcp_chatbot" / "settings.json").read_text(encoding="utf-8"))
            file_root = settings.get("file_root") or "."
        except Exception:
            file_root = "."
        db_path = str(Path(file_root) / ".agent" / "memory.db")
    _config["db_path"] = db_path
    logging.info("Using db_path: %s", db_path)
    _config["embed_base_url"] = args.embed_url
    _config["embed_model"] = args.embed_model
    try:
        import spacy
        _config["nlp"] = spacy.load("en_core_web_sm")
        logging.info("spaCy en_core_web_sm loaded.")
    except Exception as e:
        logging.warning("spaCy not available, entity boost disabled: %s", e)
        _config["nlp"] = None

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await _server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memory-retrieval",
                    server_version="0.1.0",
                    capabilities=_server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
