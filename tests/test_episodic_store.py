from pathlib import Path


def _store(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    return EpisodicStore(tmp_path / ".agent" / "memory.db")


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_db_file(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / ".agent" / "memory.db")
    assert (tmp_path / ".agent" / "memory.db").exists()


def test_init_creates_parent_dirs(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "deep" / "nested" / "memory.db")
    assert (tmp_path / "deep" / "nested" / "memory.db").exists()


def test_init_is_idempotent(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    db = tmp_path / "memory.db"
    EpisodicStore(db)
    EpisodicStore(db)  # second init should not raise or corrupt


# ── get_recent ────────────────────────────────────────────────────────────────

def test_get_recent_empty_when_no_episodes(tmp_path):
    assert _store(tmp_path).get_recent(5) == []


def test_get_recent_returns_added_episode(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess-1", "User worked on MCP client.", "agent")
    results = store.get_recent(5)
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-1"
    assert results[0]["summary"] == "User worked on MCP client."
    assert results[0]["source"] == "agent"
    assert isinstance(results[0]["created_at"], float)


def test_get_recent_returns_last_n_by_created_at(tmp_path):
    store = _store(tmp_path)
    # Use explicit created_at values to avoid sub-millisecond timestamp ties
    for i in range(5):
        store.add_episode("sess", f"Summary {i}", "agent", created_at=float(i))
    results = store.get_recent(3)
    assert len(results) == 3
    assert results[0]["summary"] == "Summary 4"
    assert results[1]["summary"] == "Summary 3"
    assert results[2]["summary"] == "Summary 2"


def test_get_recent_returns_all_when_fewer_than_n(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Only one", "agent")
    assert len(store.get_recent(10)) == 1


# ── clear_all ─────────────────────────────────────────────────────────────────

def test_clear_all_removes_episodes(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "A summary", "agent")
    store.clear_all()
    assert store.get_recent(5) == []


def test_clear_all_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.clear_all()
    store.clear_all()
    assert store.get_recent(5) == []


# ── persistence ───────────────────────────────────────────────────────────────

def test_data_survives_reconnect(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    db_path = tmp_path / "memory.db"
    EpisodicStore(db_path).add_episode("sess", "Persistent", "agent")
    results = EpisodicStore(db_path).get_recent(5)
    assert len(results) == 1
    assert results[0]["summary"] == "Persistent"


# ── WAL + FTS5 ────────────────────────────────────────────────────────────────

def test_wal_mode_enabled(tmp_path):
    import sqlite3
    from contextlib import closing
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "memory.db")
    with closing(sqlite3.connect(str(tmp_path / "memory.db"))) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_fts5_table_created(tmp_path):
    import sqlite3
    from contextlib import closing
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "memory.db")
    with closing(sqlite3.connect(str(tmp_path / "memory.db"))) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes_fts'"
        ).fetchone()
    assert row is not None


# ── remember_fact ─────────────────────────────────────────────────────────────

def test_remember_fact_returns_int_id(tmp_path):
    store = _store(tmp_path)
    fact_id = store.remember_fact("User prefers dark mode.")
    assert isinstance(fact_id, int) and fact_id > 0


def test_remember_fact_source_defaults_to_agent(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Test fact.")
    facts = store.get_key_facts()
    assert facts[0]["source"] == "agent"


def test_remember_fact_custom_source(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User stated this.", source="user")
    facts = store.get_key_facts()
    assert facts[0]["source"] == "user"


# ── update_fact ───────────────────────────────────────────────────────────────

def test_update_fact_returns_true_on_success(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("Old fact.")
    assert store.update_fact(fid, "New fact.") is True


def test_update_fact_updates_text(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("Old fact.")
    store.update_fact(fid, "New fact.")
    facts = store.get_key_facts()
    assert facts[0]["fact"] == "New fact."


def test_update_fact_returns_false_for_missing_id(tmp_path):
    store = _store(tmp_path)
    assert store.update_fact(999, "Anything.") is False


# ── forget_fact ───────────────────────────────────────────────────────────────

def test_forget_fact_returns_true_on_success(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("To be deleted.")
    assert store.forget_fact(fid) is True


def test_forget_fact_removes_row(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("To be deleted.")
    store.forget_fact(fid)
    assert store.get_key_facts() == []


def test_forget_fact_returns_false_for_missing_id(tmp_path):
    store = _store(tmp_path)
    assert store.forget_fact(999) is False


# ── get_key_facts ─────────────────────────────────────────────────────────────

def test_get_key_facts_empty_when_none(tmp_path):
    assert _store(tmp_path).get_key_facts() == []


def test_get_key_facts_returns_inserted(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Fact A.")
    facts = store.get_key_facts()
    assert len(facts) == 1
    assert facts[0]["fact"] == "Fact A."


def test_get_key_facts_ordered_by_last_accessed(tmp_path):
    # still valid: update_fact bumps last_accessed, which drives the decay score in get_key_facts
    store = _store(tmp_path)
    id1 = store.remember_fact("Fact 1.")
    store.remember_fact("Fact 2.")
    store.update_fact(id1, "Fact 1 updated.")  # bumps last_accessed for id1
    facts = store.get_key_facts()
    assert facts[0]["id"] == id1


# ── clear_all includes key_facts ──────────────────────────────────────────────

def test_clear_all_removes_key_facts(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Some fact.")
    store.clear_all()
    assert store.get_key_facts() == []


# ── search_memory ─────────────────────────────────────────────────────────────

def test_search_memory_finds_episode_via_fts5(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "MCP client Gradio development session", "agent")
    results = store.search_memory("Gradio")
    assert any(r["type"] == "episode" for r in results)
    assert any("Gradio" in r["text"] for r in results)


def test_search_memory_finds_fact_via_like(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User prefers dark mode.")
    results = store.search_memory("dark mode")
    assert any(r["type"] == "fact" for r in results)
    assert any("dark mode" in r["text"] for r in results)


def test_search_memory_matches_conversational_query(tmp_path):
    # Tokens are OR-joined: a full natural-language question matches rows sharing
    # only some keywords. Raw FTS5 implicit-AND would require every filler word
    # ("what", "about", "being") present and match nothing.
    store = _store(tmp_path)
    store.remember_fact(
        "Entities are initialized with None as their tilemap; BaseScene must assign one."
    )
    store.add_episode("sess", "Refactored tilemap assignment in base scene.", "agent")
    results = store.search_memory("And what about tilemap being None base scene and entities ?")
    assert any(r["type"] == "fact" for r in results)
    assert any(r["type"] == "episode" for r in results)


def test_search_memory_no_match_returns_empty(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Something unrelated.", "agent")
    results = store.search_memory("xyzzy_no_match_abc")
    assert results == []


def test_search_memory_result_has_required_keys(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Test episode content here.", "agent")
    results = store.search_memory("episode")
    assert len(results) > 0
    assert {"type", "id", "text", "created_at", "source", "score"} <= results[0].keys()


def test_search_memory_results_sorted_by_score(tmp_path):
    store = _store(tmp_path)
    # Episodes and facts with varying keyword density create FTS5 BM25 score spreads.
    # When episodes are appended first then facts, the concatenated list is unsorted
    # unless search_memory globally sorts by score.
    store.add_episode("s1", "Python", "agent")
    store.add_episode("s2", "Python Python Python Python Python", "agent")
    store.remember_fact("Python Python Python Python")
    store.remember_fact("Python")
    results = store.search_memory("Python", limit=10)
    assert len(results) >= 4, "need multiple results to test ordering"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores), "results must be sorted by score ascending"


def test_search_memory_respects_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.add_episode("s", f"Python tutorial episode number {i}", "agent")
    for i in range(5):
        store.remember_fact(f"Python fact number {i}")
    results = store.search_memory("Python", limit=4)
    assert len(results) <= 4, "results must not exceed limit"


# ── search_facts ──────────────────────────────────────────────────────────────

def test_search_facts_finds_fact(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User prefers dark mode.")
    results = store.search_facts("dark mode")
    assert results
    assert all(r["type"] == "fact" for r in results)
    assert any("dark mode" in r["text"] for r in results)


def test_search_facts_excludes_episodes(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Python tutorial episode about dark mode themes", "agent")
    store.remember_fact("User prefers dark mode.")
    results = store.search_facts("dark mode")
    assert results
    assert all(r["type"] == "fact" for r in results)


def test_search_facts_not_crowded_out_by_episodes(tmp_path):
    # Regression: search_memory merges episodes+facts under one shared limit, so
    # many matching episodes can evict every fact before the caller filters to
    # facts. search_facts queries only key_facts, so the fact survives.
    store = _store(tmp_path)
    for i in range(10):
        store.add_episode("s", f"Python episode number {i} about retrieval", "agent")
    store.remember_fact("Python retrieval fact worth keeping")
    results = store.search_facts("Python retrieval", limit=5)
    assert any(r["type"] == "fact" for r in results)


def test_search_facts_matches_conversational_query(tmp_path):
    # Tokens are OR-joined: a full natural-language question must still match a
    # fact sharing only some keywords. Raw FTS5 implicit-AND would require every
    # filler word ("what", "about", "being") to appear and match nothing.
    store = _store(tmp_path)
    store.remember_fact(
        "Entities are initialized with None as their tilemap; BaseScene must assign one."
    )
    results = store.search_facts("And what about tilemap being None base scene and entities ?")
    assert any(r["type"] == "fact" for r in results)


def test_search_facts_excludes_pending(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Pending unreviewed fact about widgets", status="pending")
    results = store.search_facts("widgets")
    assert results == []


def test_search_facts_respects_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.remember_fact(f"Python fact number {i}")
    results = store.search_facts("Python", limit=3)
    assert len(results) <= 3


# ── _fact_similar ─────────────────────────────────────────────────────────────

def test_fact_similar_identical_strings():
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("User prefers dark mode.", "User prefers dark mode.") is True


def test_fact_similar_near_duplicate_phrasing():
    from mcp_chatbot.memory.store import _fact_similar
    # same words, minor rephrasing — should match
    assert _fact_similar(
        "User prefers dark mode interface.",
        "User prefers dark mode interface setting.",
    ) is True


def test_fact_similar_distinct_facts():
    from mcp_chatbot.memory.store import _fact_similar
    # completely different content
    assert _fact_similar("User prefers dark mode.", "Project uses Python 3.12.") is False


def test_fact_similar_case_insensitive():
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("User Prefers Dark Mode.", "user prefers dark mode.") is True


# ── find_similar_fact ─────────────────────────────────────────────────────────

def test_find_similar_fact_empty_store_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.find_similar_fact("User prefers dark mode.") is None


def test_find_similar_fact_returns_none_for_distinct_fact(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User prefers dark mode.")
    assert store.find_similar_fact("Project uses Python 3.12.") is None


def test_find_similar_fact_returns_match_for_near_duplicate(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("User prefers dark mode interface.")
    match = store.find_similar_fact("User prefers dark mode interface setting.")
    assert match is not None
    assert match["id"] == fid
    assert "dark mode" in match["fact"]


def test_find_similar_fact_returns_dict_with_id_and_fact(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("User prefers dark mode.")
    match = store.find_similar_fact("User prefers dark mode.")
    assert isinstance(match, dict)
    assert "id" in match and "fact" in match
    assert match["id"] == fid


def test_fact_similar_numeric_difference_blocked():
    # Facts differing only by a number are treated as updates, not new facts.
    # "API timeout is 60s" is a correction of "API timeout is 30s" — use update_fact.
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("API timeout is 30 seconds.", "API timeout is 60 seconds.") is True


# ── key_facts column migration ────────────────────────────────────────────────

def test_key_facts_has_pinned_tags_status_columns(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    store = EpisodicStore(tmp_path / "memory.db")
    import sqlite3
    with sqlite3.connect(store._path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(key_facts)")}
    assert {"pinned", "tags", "status"} <= cols


def test_existing_facts_default_to_approved(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    store = EpisodicStore(tmp_path / "memory.db")
    fid = store.remember_fact("a default fact")
    import sqlite3
    with sqlite3.connect(store._path) as conn:
        row = conn.execute(
            "SELECT status, pinned FROM key_facts WHERE id=?", (fid,)
        ).fetchone()
    assert row[0] == "approved"
    assert row[1] == 0


def test_ensure_columns_upgrades_old_schema(tmp_path):
    import sqlite3
    from mcp_chatbot.memory.store import EpisodicStore
    db = tmp_path / "memory.db"
    # Create an OLD-schema key_facts table lacking the new columns.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE key_facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT NOT NULL, "
            "created_at REAL, last_accessed REAL, source TEXT)"
        )
        conn.execute(
            "INSERT INTO key_facts (fact, created_at, last_accessed, source) "
            "VALUES ('legacy fact', 0, 0, 'user')"
        )
    # Constructing the store must add the columns idempotently and not error.
    store = EpisodicStore(db)
    with sqlite3.connect(store._path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(key_facts)")}
        row = conn.execute(
            "SELECT status, pinned FROM key_facts WHERE fact='legacy fact'"
        ).fetchone()
    assert {"pinned", "tags", "status"} <= cols
    assert row[0] == "approved"  # NOT NULL DEFAULT applies to the new column
    assert row[1] == 0


# ── status / pinned filters ───────────────────────────────────────────────────

def test_get_key_facts_excludes_pending(tmp_path):
    import sqlite3
    from mcp_chatbot.memory.store import EpisodicStore
    store = EpisodicStore(tmp_path / "memory.db")
    store.remember_fact("approved fact")  # defaults to approved
    with sqlite3.connect(store._path) as conn:
        conn.execute(
            "INSERT INTO key_facts (fact, created_at, last_accessed, source, status) "
            "VALUES ('pending fact', 0, 0, 'agent', 'pending')"
        )
    facts = store.get_key_facts()
    texts = [f["fact"] for f in facts]
    assert "approved fact" in texts
    assert "pending fact" not in texts


def test_get_pinned_facts_only_pinned_approved(tmp_path):
    import sqlite3
    from mcp_chatbot.memory.store import EpisodicStore
    store = EpisodicStore(tmp_path / "memory.db")
    pid = store.remember_fact("pin me")       # approved, unpinned
    store.remember_fact("unpinned")           # approved, unpinned
    with sqlite3.connect(store._path) as conn:
        conn.execute("UPDATE key_facts SET pinned = 1 WHERE id = ?", (pid,))
    pinned = store.get_pinned_facts(20)
    texts = [f["fact"] for f in pinned]
    assert texts == ["pin me"]


def test_search_memory_excludes_pending_facts(tmp_path):
    import sqlite3
    from mcp_chatbot.memory.store import EpisodicStore
    store = EpisodicStore(tmp_path / "memory.db")
    store.remember_fact("zebra approved")  # approved, should be found
    with sqlite3.connect(store._path) as conn:
        conn.execute(
            "INSERT INTO key_facts (fact, created_at, last_accessed, source, status) "
            "VALUES ('zebra pending', 0, 0, 'agent', 'pending')"
        )
    results = store.search_memory("zebra")
    assert all(r["text"] != "zebra pending" for r in results)
    assert any(r["text"] == "zebra approved" for r in results)
